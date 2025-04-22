# paths
DET_MODEL_PATH = r'data/opt_model.pt'
REC_MODEL_NAME = r'raxtemur/trocr-base-ru'
DB_PATH = r'data/students.txt'

# detection
INPUT_RESOLUTION = (800, 800)
MIN_CONF_SCORE = 0.0

# postprocessing
DBSCAN_PARAMS = {
    'eps': 8,
    'min_samples': 1,
    'metric': 'l1',
}
MERGE_PARAMS = {
    'x_thresh': 1,
}

# comparison params
SIMILARITY_THRESH = 0.7
