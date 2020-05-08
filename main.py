# -*- encoding: utf8 -*-
import cv2
import fire
import logging
from tools.catch_face import catch_video, paint_chinese_opencv
from model.train import train_model, predict_model
from PIL import Image

FACE_LABEL = {
    0: "老居",
    1: "小居: 只想喝奶!"
}


def catch(tag):
    catch_video(tag)
    logging.info("catch_face done.")


def train():
    train_model()
    logging.info("train done.")


def predict():
    def face_predict(image):
        PIL_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return predict_model(PIL_image)

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
                label = face_predict(image)

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                frame = paint_chinese_opencv(frame, FACE_LABEL[label], (x-10, y+h+10), color)

#        cv2.imwrite("data/tmp/{}.jpg".format(int(time.time())), frame)
        return frame

    recognize_video()
    logging.info("test done.")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire()
