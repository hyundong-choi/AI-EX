import cv2
import numpy as np

image = cv2.imread('../../1.Data/tomato.png', cv2.IMREAD_GRAYSCALE)
otsu_thr, otsu_mask = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

_, contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

image_external = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)

image_internal = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image_internal, contours, i, 255, -1)

cv2.imshow("external", image_external)
cv2.imshow("internal", image_internal)
cv2.waitKey()


connectivity = 8

output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)

num_labels, labelmap, stats, centers = output

colored = np.full((image.shape[0], image.shape[1], 3), 0, np.uint8)

l = 1
img = cv2.cvtColor(otsu_mask * 255, cv2.COLOR_GRAY2BGR)
cv2.imshow('Connected components', np.hstack((img, colored)))
while True:
     k = cv2.waitKey(3)
     if k == 32:
        #for l in range(1, 5):
        if stats[l][4] > 200:
           colored[labelmap == l] = (0, 255 * l / num_labels, 255 * num_labels / l)
           cv2.circle(colored, (int(centers[l][0]), int(centers[l][1])), 5, (255, 0, 0), cv2.FILLED)
        img = cv2.cvtColor(otsu_mask * 255, cv2.COLOR_GRAY2BGR)
        l+=1
        cv2.imshow('Connected components', np.hstack((img, colored)))

        if l > 6:
            break


cv2.imshow('Connected components', np.hstack((img, colored)))
cv2.waitKey()