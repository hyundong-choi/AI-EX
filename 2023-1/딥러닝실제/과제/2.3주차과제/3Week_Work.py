# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 이미지 읽어오기
# python File과 같은 폴더에 위치한 "picture.png"파일 읽어 오기
img = imread('./picture.png')

plt.imshow(img)

plt.show()