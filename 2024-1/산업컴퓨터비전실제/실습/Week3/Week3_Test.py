import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../../1.Data/Lena.png',cv2.IMREAD_COLOR)
cv2.imshow("Color Img", img)

#Convert Gray
gray_Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Img", gray_Img)

#histogram Equalization
gray_Img_eq = cv2.equalizeHist(gray_Img)
"""
hist, bins = np.histogram(gray_Img_eq, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel("pixel value")
plt.show()
"""
cv2.imshow("Gray Img - eq", gray_Img_eq)

#Gamma Correction
gamma_gray_Img = gray_Img.astype(np.float32) / 255
gamma = 0.5
gamma_gray_Img = np.power(gamma_gray_Img, gamma)
cv2.imshow("Gamma Img", gamma_gray_Img)

#HSV
hsv_Img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h_img, s_img, v_img = cv2.split(hsv_Img)

#H -> Median FIlter
h_img = cv2.normalize(h_img, None, 0, 255, cv2.NORM_MINMAX)
h_Median_Img = cv2.medianBlur(h_img, 7)

#S -> Gaussian FIlter
s_img = cv2.normalize(s_img, None, 0, 255, cv2.NORM_MINMAX)
s_Gaussian_Img = cv2.GaussianBlur(s_img, (7, 7), 0)

#V -> Bilateral FIlter
v_img = cv2.normalize(v_img, None, 0, 255, cv2.NORM_MINMAX)
v_Bilateral_Img = cv2.bilateralFilter(v_img, -1, 0.3, 10)

cv2.imshow("H", h_img)
cv2.imshow("H_Median", h_Median_Img)
cv2.imshow("S", s_img)
cv2.imshow("S_Gaussian", s_Gaussian_Img)
cv2.imshow("V", v_img)
cv2.imshow("V_Bilateral", v_Bilateral_Img)

cv2.waitKey(0)
cv2.destroyAllWindows()