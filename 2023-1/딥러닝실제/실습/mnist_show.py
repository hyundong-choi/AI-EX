# coding: utf-8
import numpy as np
import tensorflow as tf
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()
index = 109
img = x_train[index]
label = t_train[index]
print(label)  # 2

img = img.flatten()
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)
