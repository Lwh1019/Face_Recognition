import numpy as np
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# 辅助函数定义
def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0


def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        boxes = []
        labels = []
        for obj in objects:
            boxes.append(generate_box(obj))
            labels.append(generate_label(obj))

        target = {"boxes": np.array(boxes), "labels": np.array(labels)}
        return target, len(objects)


# 加载数据
images_path = "data/images/"
annotations_path = "data/annotations/"
image_files = sorted(os.listdir(images_path))
annotation_files = sorted(os.listdir(annotations_path))

targets, numobjs = [], []
for i in range(len(image_files)):
    img_file = f"maksssksksss{i}.png"
    annotation_file = f"maksssksksss{i}.xml"
    label_path = os.path.join(annotations_path, annotation_file)

    target, numobj = generate_target(i, label_path)
    targets.append(target)
    numobjs.append(numobj)

face_images, face_labels = [], []
for i in range(len(image_files)):
    img_path = os.path.join(images_path, f"maksssksksss{i}.png")
    img = cv2.imread(img_path)

    for j in range(numobjs[i]):
        box = targets[i]['boxes'][j]
        cropped_face = img[box[1]:box[3], box[0]:box[2]]
        cropped_face = cv2.resize(cropped_face, (224, 224))
        cropped_face = img_to_array(cropped_face)
        cropped_face = preprocess_input(cropped_face)

        face_images.append(cropped_face)
        face_labels.append(targets[i]['labels'][j])

# 转换为数组
face_images = np.array(face_images, dtype="float32")
face_labels = np.array(face_labels, dtype="int")

# 标签编码和类别转换
lb = LabelEncoder()
labels = lb.fit_transform(face_labels)
labels = to_categorical(face_labels).astype("float32")

# 数据增强
aug = ImageDataGenerator(
    zoom_range=0.1,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# 构建模型
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)  # MobileNetV2的最后一个卷积层之后添加平均池化层，减少特征图的大小。
headModel = Flatten(name="flatten")(headModel)  # 将特征图展平成一维，以便连接全连接层
headModel = Dense(256, activation="relu")(headModel)  # 添加一个具有256个神经元的全连接层，用于学习更加复杂的特征表示。
headModel = Dropout(0.25)(headModel)  # 防止过拟合
headModel = Dense(3, activation="softmax")(headModel)  # 使用softmax激活函数，生成三个类别的概率分布。
model = Model(inputs=baseModel.input, outputs=headModel)

# 冻结 baseModel 的层
for layer in baseModel.layers:
    layer.trainable = False

# 初始化超参数
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# 分割数据集
(trainX, testX, trainY, testY) = train_test_split(face_images, labels, test_size=0.2, stratify=labels, random_state=42)

# 配置优化器
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练模型
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    class_weight={0: 5, 1: 1, 2: 10}
)

# 模型评估
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs))

# 保存模型
model.save('mask.h5')
