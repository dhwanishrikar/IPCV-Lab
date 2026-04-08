import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Start

# Step 2: Load Input Image 
# Note: Replace "image_path.jpg" with the actual path to your image
image = cv2.imread(r"D:\DHWANI\ENGINEERING\VI\IPCV\ipcv_lab\image.jpg")
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Step 3: Preprocessing (Convert to Grayscale)
# Checking if the image is color (3 channels) before converting
if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image.copy()

# Step 4: Create Low-Contrast Image (Mathematical Transformation)
# We multiply pixel values by 0.5 (divide by 2) to reduce dynamic range, 
# and add an offset of 50 to shift the brightness.
low_contrast = cv2.convertScaleAbs(gray, alpha=0.5, beta=50)

# Step 5: Enhancement (CLAHE)
# Create a CLAHE object with a clipLimit of 2.0 and an 8x8 tile grid size
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(low_contrast)

# Step 6: Image Smoothing (Blurring)
# Applying a 5x5 Gaussian Blur to reduce noise before segmentation
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

# Step 7: Segmentation (Otsu’s Thresholding)
# cv2.THRESH_OTSU automatically calculates the optimal threshold value.
# The 'ret' variable stores the calculated threshold value (though we don't display it here).
ret, segmented = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 8: Visualization
# Displaying the images side-by-side
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("1. Low Contrast Image")
plt.imshow(low_contrast, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("2. Enhanced (CLAHE)")
plt.imshow(enhanced, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("3. Segmented (Otsu)")
plt.imshow(segmented, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 9: Stop
