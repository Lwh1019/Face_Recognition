import os
import cv2
import dlib
import PyQt6
import random
import numpy as np
import tensorflow as tf
from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QFileDialog
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from enum import Enum
from collections import deque
from face_train import Model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

IMAGE_SIZE = 64

"人脸区域"
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
faceNet = cv2.dnn.readNetFromCaffe(configFile, modelFile)

"口罩检测模型"
mask_model = load_model('mask.h5')

"dlib 68点位模型"
dlib_model = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class HeadState(Enum):
    normal = 1
    turn_left = 2
    turn_right = 4


class MouthState(Enum):
    mouth_close = 1
    mouth_open = 2


class EyesState(Enum):
    eyes_open = 1
    eyes_close = 2


eyes_ar_thresh = 0.4
recent_eye_states = deque(maxlen=5)


def calc_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def calc_distance_shape(shape, idx_from, idx_to):
    pt1 = (shape.part(idx_from).x, shape.part(idx_from).y)
    pt2 = (shape.part(idx_to).x, shape.part(idx_to).y)
    return calc_distance(pt1, pt2)


def head_check(shape):
    left_cheek = shape.part(0)
    right_cheek = shape.part(16)
    nose_tip = shape.part(30)

    left_dist = nose_tip.x - left_cheek.x
    right_dist = right_cheek.x - nose_tip.x

    if right_dist > left_dist + 7:
        return HeadState.turn_left
    elif left_dist > right_dist + 7:
        return HeadState.turn_right

    return HeadState.normal


def mouth_check(shape):
    upper_lip = shape.part(51)
    lower_lip = shape.part(57)

    lip_distance = abs(upper_lip.y - lower_lip.y)

    if lip_distance > 15:
        return MouthState.mouth_open

    return MouthState.mouth_close


def calc_ar_value(pts):
    return (calc_distance(pts[1], pts[5]) + calc_distance(pts[2], pts[4])) / (2 * calc_distance(pts[0], pts[3]))


def eyes_check(shape):
    eye_pts_left = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
    ar_left = calc_ar_value(eye_pts_left)

    eye_pts_right = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
    ar_right = calc_ar_value(eye_pts_right)

    current_state = EyesState.eyes_open if (ar_left + ar_right > eyes_ar_thresh * 2) else EyesState.eyes_close

    recent_eye_states.append(current_state)

    if len(recent_eye_states) >= 2:
        if EyesState.eyes_close in recent_eye_states:
            return EyesState.eyes_close

    return EyesState.eyes_open


class Face(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.current_action = None
        self.cnt = None
        self.mp = [0] * 4
        self.action_timer = None
        self.shape = None
        self.captureTimer = None
        self.totalImages = None
        self.imageCount = None
        uic.loadUi('Face.ui', self)
        self.cap = cv2.VideoCapture(0)
        self.video_open = self.findChild(QtWidgets.QPushButton, 'video_open')
        self.video_close = self.findChild(QtWidgets.QPushButton, 'video_off')
        self.mask_detect = self.findChild(QtWidgets.QPushButton, 'mask_detect')
        self.dlib_detect = self.findChild(QtWidgets.QPushButton, 'dlib_detect')
        self.combox = self.findChild(QtWidgets.QComboBox, 'comboBox')
        self.start_detect = self.findChild(QtWidgets.QPushButton, 'startdetect')
        self.infoget = self.findChild(QtWidgets.QPushButton, 'infoget')
        self.htlabel = self.findChild(QtWidgets.QLabel, 'htlabel')
        self.infolabel = self.findChild(QtWidgets.QLabel, 'infolabel')
        self.lineEdit = self.findChild(QtWidgets.QLineEdit, 'lineEdit')

        self.cameraOpened = False

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateFrame)

        self.video_open.clicked.connect(self.VideoOpen)
        self.video_close.clicked.connect(self.VideoClose)
        self.mask_detect.clicked.connect(self.MaskDetect)
        self.dlib_detect.clicked.connect(self.DlibDetect)
        self.start_detect.clicked.connect(self.liveDectect)
        self.infoget.clicked.connect(self.Infoget)

        self.comboBox.addItems(["预览", "检测", "识别"])
        self.comboBox.currentIndexChanged.connect(self.moduleChange)

        self.choice = 0
        self.mask_flag = 0
        self.dlib_flag = 0
        self.face_image = None

        self.model = Model()
        self.model.load_model()
        self.label_dict = {0: 'dh', 1: 'lwh', 2: 'pfx', 13: 'Unknown'}

    def moduleChange(self, index):
        self.choice = index

    def VideoOpen(self):
        if self.cap.isOpened():
            self.timer.start()

    def VideoClose(self):
        if self.cap.isOpened():
            self.timer.stop()
            cv2.destroyAllWindows()

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            print('ERROR ON RET')
            return

        (h, w) = frame.shape[:2]

        if self.choice == 1 or self.choice == 2:
            blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104.0, 177.0, 123.0), False, False)
            faceNet.setInput(blob)
            detections = faceNet.forward()
            faces_list = []
            locations = []

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")

                    x, y = max(0, x), max(0, y)
                    x1, y1 = min(w - 1, x1), min(h - 1, y1)

                    face_frame = frame[y:y1, x:x1]
                    self.face_image = face_frame
                    if face_frame.size == 0:
                        continue
                    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                    face_frame = cv2.resize(face_frame, (224, 224))
                    face_frame = img_to_array(face_frame)
                    face_frame = np.expand_dims(face_frame, axis=0)
                    face_frame = preprocess_input(face_frame)
                    faces_list.append(face_frame)
                    locations.append((x, y, x1, y1))

            if len(faces_list) > 0:
                # 将 faces_list 转换为单个 4D 张量
                faces_input = np.vstack(faces_list)
                preds = mask_model.predict(faces_input, verbose=0)

                for (pred, (x, y, x1, y1)) in zip(preds, locations):
                    (withoutMask, mask, incorrect) = pred
                    if mask > withoutMask and mask > incorrect:
                        label = "With Mask"
                        color = (0, 255, 0)
                    elif withoutMask > mask and withoutMask > incorrect:
                        label = "Without Mask"
                        color = (0, 0, 255)
                    else:
                        label = "Wear Mask incorrectly"
                        color = (0, 255, 255)

                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask, incorrect) * 100)
                    if self.mask_flag == 1:
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                    # 使用 dlib 进行 68 点位检测

                    if self.choice == 2:
                        face_frame_detect = frame[y:y1, x:x1]
                        faceID = self.model.face_predict(face_frame_detect)
                        name = self.label_dict.get(faceID)
                        color = (0, 255, 0)
                        cv2.putText(frame, name, (x1 - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    dlib_rect = dlib.rectangle(x, y, x1, y1)
                    self.shape = dlib_predictor(frame, dlib_rect)

                    if self.dlib_flag == 1:
                        # 绘制 68 个关键点
                        for i in range(0, 68):
                            part = self.shape.part(i)
                            cv2.circle(frame, (part.x, part.y), 2, (255, 255, 0), -1)

        cv2.imshow('Video', frame)

    def MaskDetect(self):
        self.mask_flag = 1 - self.mask_flag

    def DlibDetect(self):
        self.dlib_flag = 1 - self.dlib_flag

    def liveDectect(self):
        if not self.cap.isOpened():
            print('摄像头未打开')
            return
        s = ["左摇头", "右摇头", "眨眼", "张开嘴巴"]
        self.mp = [0] * 4
        self.cnt = 0
        self.current_action = None  # 新增标志位来存储当前动作
        self.action_timer = QtCore.QTimer(self)
        self.action_timer.timeout.connect(lambda: self.on_action_timeout(s))
        self.action_timer.start()

    def on_action_timeout(self, s):
        if self.cnt >= 3:
            self.htlabel.setStyleSheet("color: red;")
            self.htlabel.setText("活体检测成功！")
            self.action_timer.stop()
            return

        if self.current_action is not None:  # 如果当前动作还未完成，则不执行新的动作
            return

        try:
            pos = random.randint(0, 3)
            while self.mp[pos]:
                pos = random.randint(0, 3)
            self.mp[pos] = 1
            self.current_action = pos  # 设置当前动作
            self.htlabel.setText(s[pos])

            QtCore.QTimer.singleShot(3000, lambda: self.check_action(pos))
        except Exception as e:
            print(e)

    def check_action(self, pos):
        head_state = head_check(self.shape)
        eyes_state = eyes_check(self.shape)
        mouth_state = mouth_check(self.shape)

        action_matched = False

        if pos == 0 and head_state == HeadState.turn_left:
            action_matched = True
        elif pos == 1 and head_state == HeadState.turn_right:
            action_matched = True
        elif pos == 2 and eyes_state == EyesState.eyes_close:
            action_matched = True
        elif pos == 3 and mouth_state == MouthState.mouth_open:
            action_matched = True

        if action_matched:
            self.htlabel.setText("动作匹配成功！")
            self.action_timer.start(1000)
            self.cnt += 1
        else:
            self.htlabel.setText("动作匹配失败，重新开始...")
            self.action_timer.start(1000)
            self.mp[pos] = 0

        # 完成当前动作后，重置标志位以允许下一个动作
        self.current_action = None

    def Infoget(self):
        if not self.cap.isOpened():
            print('摄像头未打开')
            return
        name = self.lineEdit.text()
        if not name:
            print('请输入名字')
            return

        dir_path = f"face_data/{name}"
        os.makedirs(dir_path, exist_ok=True)

        self.imageCount = 0
        self.totalImages = 1000

        self.captureTimer = QtCore.QTimer(self)
        self.captureTimer.timeout.connect(lambda: self.captureFaceImage(dir_path))
        self.captureTimer.start(500)

    def captureFaceImage(self, dir_path):
        if self.imageCount >= self.totalImages:
            self.captureTimer.stop()
            print(f"采集完成")
            return

        if self.face_image is not None and self.face_image.size != 0:
            image_path = os.path.join(dir_path, f"{self.imageCount}.jpg")
            cv2.imwrite(image_path, self.face_image)
            self.imageCount += 1
            self.infolabel.setText(f"已采集 {self.imageCount}/{self.totalImages} 张人脸照片")
        else:
            print("未检测到有效的人脸图像")


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    win = Face()
    win.show()
    sys.exit(app.exec())
