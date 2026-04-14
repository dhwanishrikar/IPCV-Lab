import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

low_contrast = gray // 2 + 50

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(low_contrast)

blur = cv2.GaussianBlur(enhanced, (5,5), 0)
_, segmented = cv2.threshold(blur, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')

plt.subplot(2,2,2)
plt.title("Low Contrast")
plt.imshow(low_contrast, cmap='gray')

plt.subplot(2,2,3)
plt.title("Enhanced (CLAHE)")
plt.imshow(enhanced, cmap='gray')

plt.subplot(2,2,4)
plt.title("Segmented")
plt.imshow(segmented, cmap='gray')

plt.tight_layout()
plt.show()
