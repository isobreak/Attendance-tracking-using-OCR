from fastapi import FastAPI, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
from src.processing import Pipeline, Detector, Recogniser
from src.constants import DET_MODEL_PATH, REC_MODEL_NAME, INPUT_RESOLUTION, MIN_CONF_SCORE, DB_PATH
import transformers
from transformers.utils import logging


logging.set_verbosity(transformers.logging.ERROR)

det = Detector(DET_MODEL_PATH, 'cpu', INPUT_RESOLUTION, MIN_CONF_SCORE, True)
try:
    rec = Recogniser(REC_MODEL_NAME, 'cuda:0', True)
except:
    rec = Recogniser(REC_MODEL_NAME, 'cpu', True)
pipe = Pipeline(det, rec)
app = FastAPI()

with open(DB_PATH, 'r', encoding='utf-8') as f:
    default_acceptable_names = f.read()
    default_acceptable_names = [[x for x in full_name.split()] for full_name in default_acceptable_names.split('\n')]


@app.post("/")
def predict(file: UploadFile):
    img = Image.open(BytesIO(file.file.read()))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    np_rgb = np.asarray(img)

    res = pipe.predict(np_rgb, default_acceptable_names)

    return res
