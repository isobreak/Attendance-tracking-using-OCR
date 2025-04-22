import os

DATA_PATH = r'../../data/train_data/train_segmentation'
ANNOT_PATH = os.path.join(DATA_PATH, 'annotations.json')
ANNOT_EXT_PATH = os.path.join(DATA_PATH, 'annotations_extended.json')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')

INPUT_RESOLUTION = (800, 800)
TEST_CENTER_CROP_RESOLUTION = (1960, 1960)

# augmentation params
BBOX_PARAMS = {
    'min_visibility': 0.25,
}
MIN_MAX_HEIGHT = (1600, 3000)

