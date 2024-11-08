import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from load_facedata import load_dataset, resize_image, IMAGE_SIZE, label_dict
from tensorflow.keras.utils import to_categorical


class Dataset:
    def __init__(self, path_name):
        # 初始化训练集、验证集、测试集
        self.train_images = None
        self.train_labels = None

        self.valid_images = None
        self.valid_labels = None

        self.test_images = None
        self.test_labels = None

        self.path_name = path_name
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=3):
        images, labels = load_dataset(self.path_name)

        # 随机划分训练集和验证集
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        # 根据Keras要求的维度顺序重组图片数据
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

        # 输出训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')

        # 将类别向量转换为二进制类别矩阵
        train_labels = to_categorical(train_labels, nb_classes)
        valid_labels = to_categorical(valid_labels, nb_classes)
        test_labels = to_categorical(test_labels, nb_classes)

        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        # 将其归一化
        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None
        self.label_to_name = {0: 'dh', 1: 'lwh', 2: 'pfx'}

    # 建立模型
    def build_model(self, dataset, nb_classes=3):
        # 构建一个空的神经网络，它是一个线性堆栈模型，每种层都可以依次被加入。专业名词为序贯模型或线性堆叠模型
        self.model = Sequential()

        # 以下代码将按照顺序向CNN网络模型中添加各层，一次add就是一个网络层
        self.model.add(Conv2D(32, (3, 3), padding='same',
                              input_shape=dataset.input_shape))  # 2D卷积层
        self.model.add(Activation('relu'))  # 激活层

        self.model.add(Conv2D(32, (3, 3)))  # 2D卷积层
        self.model.add(Activation('relu'))  # 激活层
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 池化层
        self.model.add(Dropout(0.25))  # Dropout层

        self.model.add(Conv2D(64, (3, 3), padding='same'))  # 2D卷积层
        self.model.add(Activation('relu'))  # 激活层
        self.model.add(Conv2D(64, (3, 3)))  # 2D卷积层
        self.model.add(Activation('relu'))  # 激活层
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 池化层
        self.model.add(Dropout(0.25))  # Dropout层

        self.model.add(Flatten())  # Flatten层
        self.model.add(Dense(512))  # Dense层，全连接层
        self.model.add(Activation('relu'))  # 激活层
        self.model.add(Dropout(0.5))  # Dropout层
        self.model.add(Dense(nb_classes))  # Dense层，全连接层
        self.model.add(Activation('softmax'))  # 分类层，输出层

        self.model.summary()

    # 训练模型
    def train(self, dataset, batch_size=20, nb_epoch=20, data_augmentation=True):
        sgd = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)  # 使用更新后的参数名
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        # 不使用数据增强，所有图像将被用作训练样本并提升网络的准确性、解锁、加载等参数方法调用训练
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,  # 更新 nb_epoch 为 epochs
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)

        # 使用实时数据增强
        else:
            print("使用实时数据增强生成器")
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False
            )

            datagen.fit(dataset.train_images)

            train_data_gen = datagen.flow(dataset.train_images, dataset.train_labels, batch_size=batch_size)

            for epoch in range(nb_epoch):
                print(f"Epoch {epoch + 1}/{nb_epoch}")
                self.model.fit(train_data_gen,
                               steps_per_epoch=len(dataset.train_images) // batch_size,
                               validation_data=(dataset.valid_images, dataset.valid_labels))
                train_data_gen.reset()  # 每轮结束后重置生成器

    MODEL_PATH = 'face_model.h5'

    def save_model(self):
        self.model.save(self.MODEL_PATH)

    def load_model(self):
        self.model = load_model(self.MODEL_PATH)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def face_predict(self, image):
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致，应该是 IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 针对1张图片进行预测
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        result = self.model.predict(image)
        np.set_printoptions(suppress=True, precision=10)
        print('result:', result)

        max_prob = np.max(result)
        result_class = result.argmax(axis=-1)[0]
        if max_prob >= 0.75:
            predicted_name = self.label_to_name.get(result_class)  # 获取对应的名字
            print(f"Predicted: {predicted_name}")
            return result_class
        else:
            print(f"Predicted: Unknown")
            return 13


if __name__ == '__main__':
    dataset = Dataset('./face_data/')
    dataset.load()

    '''
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model()
    '''
    model = Model()
    model.load_model()
    model.evaluate(dataset)



