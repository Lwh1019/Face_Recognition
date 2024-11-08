import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# 加载 DNN 人脸检测模型
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
faceNet = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# 加载口罩检测模型
mask_model = load_model("mask.h5")

# 设置视频捕捉设备
video_capture = cv2.VideoCapture(0)

while True:
    # 逐帧捕获
    ret, frame = video_capture.read()
    if not ret:
        break

    # 获取图像的高宽
    (h, w) = frame.shape[:2]

    # 将图像转换为 blob，并传入人脸检测模型
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces_list = []
    locations = []

    # 遍历检测到的每个检测框
    for i in range(0, detections.shape[2]):
        # 提取置信度
        confidence = detections[0, 0, i, 2]

        # 设置阈值，过滤低置信度的检测结果
        if confidence > 0.5:
            # 计算检测框的坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # 确保坐标在图像范围内
            x, y = max(0, x), max(0, y)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)

            # 提取人脸区域并进行预处理
            face_frame = frame[y:y1, x:x1]
            if face_frame.size == 0:
                continue
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = preprocess_input(face_frame)

            # 将预处理的人脸图像和位置存入列表
            faces_list.append(face_frame)
            locations.append((x, y, x1, y1))

    # 如果检测到人脸，进行口罩检测
    if len(faces_list) > 0:
        # 将 faces_list 转换为单个 4D 张量
        faces_input = np.vstack(faces_list)
        preds = mask_model.predict(faces_input)

        # 处理每个预测结果
        for (pred, (x, y, x1, y1)) in zip(preds, locations):
            (withoutMask, mask, incorrect) = pred
            if mask > withoutMask and mask > incorrect:
                label = "Mask"
                color = (0, 255, 0)  # 绿色
            elif withoutMask > mask and withoutMask > incorrect:
                label = "No Mask"
                color = (0, 0, 255)  # 红色
            else:
                label = "Incorrect Mask"
                color = (0, 255, 255)  # 黄色

            # 绘制标签和边框
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask, incorrect) * 100)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)

    # 显示结果帧
    cv2.imshow('Video', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕捉设备和关闭窗口
video_capture.release()
cv2.destroyAllWindows()
