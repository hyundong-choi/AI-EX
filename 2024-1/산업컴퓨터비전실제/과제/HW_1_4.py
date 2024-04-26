import cv2
import numpy as np

image = cv2.imread('../../1.Data/Lena.png', cv2.IMREAD_GRAYSCALE)

binary_method = input("이진화 방법 (otsu : OT / adaptive median : AM): ").upper()
morphology_operation = input("모폴로지 연산 (erosion : E, dilation : D, opening : O, closing : C): ").upper()
iterations = int(input("모폴로지 연산 적용 횟수 : "))

# 이진화
if binary_method == 'OT':
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
elif binary_method == 'AM':
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# 모폴로지 연산 수행
kernel = np.ones((3, 3), np.uint8)
if morphology_operation == 'E':
    result = cv2.erode(binary_image, kernel, iterations=iterations)
elif morphology_operation == 'D':
    result = cv2.dilate(binary_image, kernel, iterations=iterations)
elif morphology_operation == 'O':
    result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
elif morphology_operation == 'C':
    result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

# 결과 출력
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Morphology Result', result)
cv2.waitKey()
cv2.destroyAllWindows()