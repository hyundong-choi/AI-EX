import cv2
import numpy as np
import matplotlib.pyplot as plt

color = cv2.imread('../../1.Data/Lena.png', cv2.IMREAD_COLOR)
color_Hist = color.copy()
color_hsv = color.copy()

channel = input("Enter the channel to perform histogram equalization (R/G/B): ").upper()

if channel == "R":
    hist, bins = np.histogram(color[..., 2], 256, [0, 255])
    color_Hist[..., 2] = cv2.equalizeHist(color[..., 2])
    plt.title('Red Histogram')
elif channel == "G":
    hist, bins = np.histogram(color[..., 1], 256, [0, 255])
    color_Hist[..., 1] = cv2.equalizeHist(color[..., 1])
    plt.title('Green Histogram')
elif channel == "B":
    hist, bins = np.histogram(color[..., 0], 256, [0, 255])
    color_Hist[..., 0] = cv2.equalizeHist(color[..., 0])
    plt.title('Blue Histogram')

plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')
plt.show()

cv2.imshow('original color', color)

if channel == "R":
    cv2.imshow('equalized Red color', color_Hist)
elif channel == "G":
    cv2.imshow('equalized Green color', color_Hist)
elif channel == "B":
    cv2.imshow('equalized Blue color', color_Hist)

hsv = cv2.cvtColor(color_hsv, cv2.COLOR_BGR2HSV)

hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('equalized HSV V-channel', color_eq)
cv2.waitKey()
cv2.destroyAllWindows()