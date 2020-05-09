# -*- encoding: utf8 -*-
import fire
import logging
from face.catch_face import catch_video
from face.recognize_face import recognize_video
from model.train import train_model


def catch(tag):
    catch_video(tag)
    logging.info("catch_face done.")


def train():
    train_model()
    logging.info("train done.")


def predict():
    recognize_video()
    logging.info("test done.")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire()
