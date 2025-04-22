import os
import cv2
import torch
from src.app.src.processing import Detector, Recogniser, find_clusters, merge_bboxes, visualize


def main():
    path = r'../../data/test_data/images'
    save_path = r'../../data/results/recognition/res.txt'

    det_path = '../../data/models/opt_model.pt'
    rec_name = 'raxtemur/trocr-base-ru'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    detector = Detector(det_path, 'cpu')
    recogniser = Recogniser(rec_name, device)

    corpus = []
    for img_name in os.listdir(path):
        image = cv2.imread(os.path.join(path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # processing
        bboxes = detector.predict(image)
        clusters = find_clusters(bboxes)
        clusters = [merge_bboxes(cluster) for cluster in clusters]
        img_texts = []
        for cluster in clusters:
            cluster_texts = recogniser.predict(image, cluster)
            img_texts.append(cluster_texts)
        # visualization
        res = visualize(image, clusters, img_texts)
        cv2.imshow('res', res)
        cv2.waitKey(1)

        corpus.append(img_texts)
    corpus_str = ';;;;;'.join([';;;;'.join([';;;'.join(cluster) for cluster in img]) for img in corpus])

    print(corpus_str)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(corpus_str)


if __name__=="__main__":
    main()