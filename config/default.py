# -*- encoding: utf8 -*-
import os

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

DATA_TMP = os.path.join(PROJECT_PATH, "data/tmp")
DATA_TRAIN = os.path.join(PROJECT_PATH, "data/train")
DATA_TEST = os.path.join(PROJECT_PATH, "data/test")
DATA_MODEL = os.path.join(PROJECT_PATH, "data/model")

DEFAULT_MODEL = "retrain.pkl"

BATCH_SIZE = 10
EPOCHS = 3
LR = 0.01
