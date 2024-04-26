import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../../1.Data/Lena.png', 0).astype(np.float32) / 255

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
print('FFT shape:', fft.shape)
print('FFT data type:', fft.dtype)

shifted = np.fft.fftshift(fft, axes=[0, 1])

magnitude = cv2.magnitude(shifted[:,:,0], shifted[:,:,1])
print(magnitude.shape)
magnitude = np.log(magnitude)

plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.tight_layout()
plt.show()