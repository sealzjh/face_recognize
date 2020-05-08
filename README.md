# Face Recognize with Pytorch CNN

## install
* $ mkvirtualenv face_recognize
* $ pip install -r requirements.txt

## catch face data for train
* $ python main.py catch 0 # one man face
* $ python main.py catch 1 # the other man

## train
* $ python main.py train

## recognize face
* $ python main.py predict