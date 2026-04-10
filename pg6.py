# 30 - 03 - 2026

import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r'D:\4SF23CI052 - IPCV\wall-39.jpg',0)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

gaussian = cv2.GaussianBlur(gray, (25, 25), 0)                 
median = cv2.medianBlur(gray, 25)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian, alpha=2)             

plt.figure(figsize=(16, 9))

plt.subplot(2, 2, 1)
plt.imshow(gray,cmap="gray")
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gaussian,cmap='gray')
plt.title('Gaussian Blur, Kernal=25')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(median,cmap='gray')
plt.title('Median Blur, Kernal=25')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Filter, Alpha=2')
plt.axis('off')

plt.tight_layout()
plt.show()
