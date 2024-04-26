import cv2
import numpy as np

image = cv2.imread('../../1.Data/Lena.png', 0).astype(np.float32) / 255

noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 1)

gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)

median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)

bilateral = cv2.bilateralFilter(noised, -1, 0.3, 10)

_abs_Gaussian = cv2.absdiff(image, gauss_blur)
_abs_Median = cv2.absdiff(image, median_blur.astype(np.float32) / 255)
_abs_Bilateral = cv2.absdiff(image, bilateral)

cv2.imshow('Ori Img', image)
cv2.imshow('Noise Img', noised)
cv2.imshow('Gaussian Img', gauss_blur)
cv2.imshow('Median Img', median_blur)
cv2.imshow('Bilateral Img', bilateral)
cv2.imshow('abs Gaussian Img', _abs_Gaussian)
cv2.imshow('abs Median Img', _abs_Median)
cv2.imshow('abs Bilateral Img', _abs_Bilateral)
cv2.waitKey()
cv2.destroyAllWindows()