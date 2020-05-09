# -*- encoding: utf8 -*-
import cv2
from model.train import predict_model
import numpy as np
from PIL import Image, ImageFont, ImageDraw


FACE_LABEL = {
    0: "老居",
    1: "小居: 只想喝奶!"
}


def recognize_video(window_name='face recognize', camera_idx=0):
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(camera_idx)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        catch_frame = catch_face(frame)
        cv2.imshow(window_name, catch_frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def catch_face(frame):
    classfier = cv2.CascadeClassifier(
        "/Users/alan/.virtualenvs/kepler/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    color = (0, 255, 0)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(face_rects) > 0:
        for face_rects in face_rects:
            x, y, w, h = face_rects
            image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            PIL_image = cv2pil(image)
            label = predict_model(PIL_image)

            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            frame = paint_chinese_opencv(frame, FACE_LABEL[label], (x-10, y+h+10), color)

#        cv2.imwrite("data/tmp/{}.jpg".format(int(time.time())), frame)
    return frame


def cv2pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def paint_chinese_opencv(im, chinese, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('/Library/Fonts/Songti.ttc', 20)
    fillColor = color
    position = pos
    if not isinstance(chinese, unicode):
        chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img