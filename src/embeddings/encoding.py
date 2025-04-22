import os
import cv2
import torch
import numpy as np
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from torchvision.transforms import v2
import onnxruntime as ort


class Detector:
    """
    Faster R-CNN model trained to detect words
    """
    def __init__(self, path: str, device: str = 'cpu',
                 model_resolution: tuple[int, int] = (800, 800), min_conf_score: float = 0, verbose: bool = False):
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


def main():
    path = r'../../data/test_data/images'

    detector_path = r'../../data/models/opt_model.pt'
    encoder_path = '../../data/models/raxtemur/encoder_model.onnx'

    detector = Detector(detector_path)

    providers = [
        'CUDAExecutionProvider'
    ]
    processor = TrOCRProcessor.from_pretrained('raxtemur/trocr-base-ru')
    session_encoder = ort.InferenceSession(encoder_path, providers=providers)

    all_embeddings = []
    for img_name in os.listdir(path):
        image = cv2.imread(os.path.join(path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # processing
        bboxes = detector.predict(image)
        images = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = [int(el) for el in bbox]
            images.append(image[y1:y2, x1:x2, :])

        pix_val = processor(images=images, return_tensors='pt').pixel_values
        pix_val = pix_val.numpy()
        encoded_list = session_encoder.run(['last_hidden_state'], {'pixel_values': pix_val})
        last_hidden_states = encoded_list[0]

        all_embeddings.append(last_hidden_states)
        print(last_hidden_states.shape)
    res_embeddings = np.concatenate(all_embeddings, axis=0)
    print(res_embeddings.shape)


if __name__=="__main__":
    main()