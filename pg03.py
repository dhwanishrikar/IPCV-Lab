import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

image = cv2.imread("path_to_your_image.jpg") 

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobelx, sobely)

sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(blur, n_points, radius, method="uniform")

plt.figure(figsize=(12, 8))
plt.subplot(2,2,1)
plt.title("1. Original Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(2,2,2)
plt.title("2. Gaussian Blur (Noise Reduction)")
plt.imshow(blur, cmap="gray")
plt.axis("off")

plt.subplot(2,2,3)
plt.title("3. Sobel Edge Magnitude")
plt.imshow(sobel_norm, cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("4. LBP Texture Features")
plt.imshow(lbp, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
