import json
import numpy as np
import cv2

from constants import *


def parse_json(data, level=0, indent='      '):
    """
    Prints JSON structure
    Args:
        data: dict with JSON
        level: basic indent counter
        indent: indent
    """

    if type(data) is dict:
        for key in data.keys():
            print(indent * level, key)
            parse_json(data[key], level + 1)
    elif type(data) is list:
        print(indent * level + 'list:')
        parse_json(data[0], level + 1)
    else:
        print(indent * level, type(data))


def main():
    with open(ANNOT_PATH, 'r') as f:
        data_js = json.load(f)

        data = {image['id']: {'img_name': image['file_name'],
                              'height': image['height'],
                              'width': image['width'],
                              'bboxes': [],
                              'segs': [],
        } for image in data_js['images']}
        for annot in data_js['annotations']:
            data[annot['image_id']]['segs'].append(annot['segmentation'][0])
            data[annot['image_id']]['bboxes'].append(annot['bbox'])
        parse_json(data)
        for key in data.keys():
            contours = []
            for seg in data[key]['segs']:
                cont = np.array([[seg[2*i], seg[2*i+1]] for i in range(len(seg) // 2)], dtype=np.int32)
                contours.append(cont)
            img = cv2.imread(os.path.join(IMAGES_PATH, data[key]['img_name']))

            for i in range(len(data[key]['bboxes'])):
                points = data[key]['bboxes'][i]
                x, y, w, h = [int(t) for t in points]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('img', cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3)))
            cv2.waitKey(1000)


if __name__ == "__main__":
    main()