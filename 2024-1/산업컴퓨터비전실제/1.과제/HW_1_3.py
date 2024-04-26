import cv2
import numpy as np

radius1 = int(input("첫 번째 원의 반지름을 입력하세요: "))
radius2 = int(input("두 번째 원의 반지름을 입력하세요: "))

# read input and convert to grayscale
img = cv2.imread('../../1.Data/Lena.png', cv2.IMREAD_GRAYSCALE)

# do dft saving as complex output
dft = np.fft.fft2(img, axes=(0,1))

# apply shift of origin to center of image
dft_shift = np.fft.fftshift(dft)

# generate spectrum from magnitude image (for viewing only)
mag = np.abs(dft_shift)
spec = np.log(mag) / 20

if radius1 > radius2 :
    temp = radius1
    radius2 = radius1
    radius1 = temp

mask1 = np.zeros_like(img)
cy = mask1.shape[0] // 2
cx = mask1.shape[1] // 2
cv2.circle(mask1, (cx,cy), radius1, (255,255,255), -1)[0]

mask2 = np.zeros_like(img)
cy = mask2.shape[0] // 2
cx = mask2.shape[1] // 2
cv2.circle(mask2, (cx,cy), radius2, (255,255,255), -1)[0]

final_mask = mask2 - mask1

dft_shift_masked_final = np.multiply(dft_shift, final_mask) / 255

back_ishift = np.fft.ifftshift(dft_shift)
back_ishift_masked_final = np.fft.ifftshift(dft_shift_masked_final)

img_back = np.fft.ifft2(back_ishift, axes=(0,1))
img_filtered_final = np.fft.ifft2(back_ishift_masked_final, axes=(0,1))

img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
img_filtered_final = np.abs(img_filtered_final).clip(0,255).astype(np.uint8)


cv2.imshow("Original", img)
cv2.imshow("Spectrum", spec)
cv2.imshow("final_mask", final_mask)
cv2.imshow("Original DFT/IFT trans", img_back)
cv2.imshow("Img_filtered_final", img_filtered_final)
cv2.waitKey()
cv2.destroyAllWindows()