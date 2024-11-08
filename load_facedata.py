import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 128


def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)

    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left

    BLACK = (0, 0, 0)
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width))


images = []
labels = []

# 使用字典为每个文件夹分配一个标签
label_dict = {}


def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            # 如果文件夹未在标签字典中，给它分配一个新标签
            if dir_item not in label_dict:
                label_dict[dir_item] = len(label_dict)
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                labels.append(label_dict[os.path.basename(path_name)])  # 根据文件夹分配标签

    return images, labels


def load_dataset(path_name):
    images, labels = read_path(path_name)
    images = np.array(images)
    labels = np.array(labels)
    print(images.shape)

    return images, labels


if __name__ == '__main__':
    images, labels = load_dataset('./face_data/')
    print("标签字典:", label_dict)  # 输出标签字典，以确认每个人物的标签编号
