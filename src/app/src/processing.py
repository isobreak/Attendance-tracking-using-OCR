import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torchvision.transforms import v2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from Levenshtein import distance
from src.constants import DBSCAN_PARAMS, MERGE_PARAMS, MIN_CONF_SCORE, SIMILARITY_THRESH


class Detector:
    """
    Faster R-CNN model trained to detect words
    """
    def __init__(self, path: str, device: str = 'cpu',
                 model_resolution: tuple[int, int] = (800, 800), min_conf_score: float = MIN_CONF_SCORE, verbose: bool = False):
        self.model = torch.load(path, weights_only=False).to(device).eval()
        self.model_resolution = model_resolution
        self.min_conf_score = min_conf_score
        if verbose:
            print(f'Detection model "{path}" is now on "{device}"')

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
            Detect bounding boxes
        Args:
            image: RGB image with shape (resolution[0], resolution[1], 3)

        Returns:
            bounding boxes coordinates corresponding to original (input) resolution
        """
        # calculate scaling vector
        IMG_HEIGHT, IMG_WIDTH, _ = image.shape
        BBOX_IMG_H, BBOX_IMG_W = self.model_resolution
        x_scale = IMG_WIDTH / BBOX_IMG_W
        y_scale = IMG_HEIGHT / BBOX_IMG_H
        scale_vector = [x_scale, y_scale, x_scale, y_scale]

        # preprocessing
        preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(self.model_resolution),
        ])
        input = preprocess(image)
        input = input.unsqueeze(0)

        # prediction
        prediction = self.model(input)
        boxes = prediction[0]['boxes'].detach().to('cpu').numpy()
        scores = prediction[0]['scores'].detach().to('cpu').numpy()
        boxes = boxes[scores > self.min_conf_score]

        boxes = boxes * scale_vector

        return boxes


class Recogniser:
    def __init__(self, model_name: str = 'raxtemur/trocr-base-ru', device: str = 'cpu', verbose: bool = False):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device=device).eval()
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.device = device
        if verbose:
            print(f'Recognition model "{model_name}" is now on "{device}"')

    def predict(self, image: np.ndarray, bboxes: np.ndarray) -> list[str]:
        """
        Processes given image in specified areas using TrOCR
        Args:
            image: RGB image of shape (H, W, 3) with any resolution
            bboxes: numpy array of shape (n_samples, 4)

        Returns:
            list of recognised texts
        """
        # crop ROIs from original image based on scaled bboxes
        images = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(el) for el in bbox]
            images.append(image[y1:y2, x1:x2, :])
        pix_val = self.processor(images=images, return_tensors='pt').pixel_values.to(device=self.device)
        gen_ids = self.model.generate(pix_val, num_beams=1)
        gen_texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        return gen_texts


class Pipeline:
    def __init__(self, detector: Detector, recogniser: Recogniser):
        self.detector = detector
        self.recogniser = recogniser

    def predict(self, image: np.ndarray, acceptable_names: list[list[str]]) -> dict:
        """
        Finds present students based on image and acceptable names
        Args:
            image: RGB image of shape (H, W, 3) with any resolution
            acceptable_names: list of acceptable names

        Returns:
            Dictionary with 'ids', 'scores' and 'full_names' that have been found
        """
        # processing
        bboxes = self.detector.predict(image)
        clusters = self._find_clusters(bboxes)
        clusters = [self._merge_bboxes(cluster) for cluster in clusters]
        img_texts = []
        for cluster in clusters:
            cluster_texts = self.recogniser.predict(image, cluster)
            img_texts.append(cluster_texts)

        result = self._get_present_students_info(acceptable_names, img_texts)

        return result

    def _merge_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        """
        Merge boxes in a given list based on their position
        Args:
            bboxes: list of bboxes of shape (n_samples, 4) in XYXY format

        Returns:
            merged bboxes (n_after_merge, 4)
        """

        def are_neighbours(a: np.ndarray, b: np.ndarray, thresh: int) -> bool:
            """Check whether a and b should be merged during postprocessing stage"""
            if a[0] < b[0]:
                left = a
                right = b
            else:
                left = b
                right = a

            if right[0] - left[2] < thresh:
                return True

            return False

        def get_merged_box(a):
            """Returns merged bbox based on a given list of bboxes"""
            x1 = min([bbox[0] for bbox in a])
            x2 = max([bbox[2] for bbox in a])
            y1 = sum([bbox[1] for bbox in a]) / len(a)
            y2 = sum([bbox[3] for bbox in a]) / len(a)

            return np.array([x1, y1, x2, y2], dtype=np.uint32)

        unprocessed = [bboxes[i] for i in range(len(bboxes))]
        processed = []
        while len(unprocessed) > 1:
            neighbours = [unprocessed[0]]
            for i in range(len(unprocessed) - 1, 0, -1):
                if are_neighbours(unprocessed[0], unprocessed[i], MERGE_PARAMS['x_thresh']):
                    neighbours.append(unprocessed[i])
                    del unprocessed[i]

            if len(neighbours) > 1:
                merged = get_merged_box(neighbours)
                unprocessed.append(merged)
            else:
                processed.append(unprocessed[0])
            del unprocessed[0]

        processed.append(unprocessed[0])
        res = np.vstack(processed)

        return res


    def _find_clusters(self, bboxes: np.ndarray) -> list[np.ndarray]:
        """
        Finds groups of bboxes representing the same student name based on avg_y of bbox
        Args:
            bboxes: numpy array of shape (n_bboxes, 4) in XYXY format

        Returns:
            clusters
        """
        clustering = DBSCAN(**DBSCAN_PARAMS)
        y1 = bboxes[:, 1]
        y2 = bboxes[:, 3]
        pos = (y1 + y2) / 2

        clustering.fit(pos.reshape(-1, 1))
        boxes_clusters = []
        for label in np.unique(clustering.labels_):
            boxes_i = bboxes[clustering.labels_ == label, :]

            # sorting bboxes based on x1 value
            indices = boxes_i[:, 0].argsort()
            sorted_bboxes = boxes_i[indices, :]
            boxes_clusters.append(sorted_bboxes)

        return boxes_clusters

    def _get_present_students_info(self, acceptable_names: list[list[str]], recognised_clusters: list[list[str]]) -> dict:
        """
        Finds present students based on recognised clusters and acceptable names
        Args:
            acceptable_names: list of acceptable names
            recognised_clusters: list of recognised clusters

        Returns:
            Dictionary with 'ids', 'scores' and 'full_names' that have been found
        """
        # preprocessing
        acceptable_names = [[name_part.lower() for name_part in name] for name in acceptable_names]
        recognised_clusters = [[token.lower() for elem in cluster for token in elem.split()] for cluster in
                               recognised_clusters]

        mat_outer = np.zeros((len(acceptable_names), len(recognised_clusters)), dtype=np.float32)
        for i, name in enumerate(acceptable_names):
            for j, cluster in enumerate(recognised_clusters):
                mat_inner = np.zeros((len(name), len(cluster)), dtype=np.float32)
                at_least_one = False
                for m, name_part in enumerate(name):
                    for n, token in enumerate(cluster):
                        d = distance(token, name_part) / len(name_part)
                        if 1 - d > SIMILARITY_THRESH:
                            mat_inner[m, n] = 1 - d
                            at_least_one = True
                if at_least_one:
                    while mat_inner.shape[1] and mat_inner.shape[0] and np.max(mat_inner):
                        i_0, i_1 = np.unravel_index(np.argmax(mat_inner), mat_inner.shape)
                        mat_outer[i, j] += mat_inner[i_0, i_1]
                        mat_inner = np.delete(mat_inner, i_0, 0)
                        mat_inner = np.delete(mat_inner, i_1, 1)
                else:
                    continue

        # get recognised ids
        recognised_ids = []
        scores = []
        while mat_outer.shape[1] and mat_outer.shape[0] and np.max(mat_outer):
            i, j = np.unravel_index(np.argmax(mat_outer), mat_outer.shape)
            recognised_ids.append(i.item())
            scores.append(mat_outer[i, j].item())
            mat_outer[:, j] = np.zeros(mat_outer.shape[0], dtype=np.float32)
            mat_outer[i, :] = np.zeros(mat_outer.shape[1], dtype=np.float32)

        return {
            'ids': recognised_ids,
            'scores': scores,
            'full_names': [acceptable_names[k] for k in recognised_ids],
        }
