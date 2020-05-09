# -*- coding: utf-8 -*-
import time
import cv2
import os

from config import DATA_TRAIN
from face.utils import makedir_exist_ok


def catch_video(tag, window_name='catch face', camera_idx=0):
    """
    get video from camera
    :return:
    """
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(camera_idx)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        catch_face(frame, tag)
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def catch_face(frame, tag):
    """
    use opencv classifier catch face
    :param frame:
    :return:
    """
    classfier = cv2.CascadeClassifier("/Users/alan/.virtualenvs/face_recognize/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    color = (0, 255, 0)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    num = 1
    if len(face_rects) > 0: #大于0则检测到人脸
        for face_rects in face_rects:
            x, y, w, h = face_rects
            image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            save_face(image, tag, num)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            num += 1


def save_face(image, tag, num):
    """
    save face tmp dir
    """
    makedir_exist_ok(os.path.join(DATA_TRAIN, str(tag)))
    img_name = os.path.join(DATA_TRAIN, str(tag), '{}_{}.jpg'.format(int(time.time()), num))
    cv2.imwrite(img_name, image)

