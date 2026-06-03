import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\DHWANI\ENGINEERING\VI\IPCV\ipcv_lab\image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img_rgb.shape[:2]  

tx, ty = 200, 150
T = np.float32([[1, 0, tx],[0, 1, ty]])
translated = cv2.warpAffine(img_rgb, T, (w+tx, h+ty)) 


angle = 45
R = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)

corners = np.float32([[0,0],[w,0],[0,h],[w,h]]).reshape(-1,1,2)
new_corners = cv2.transform(corners, R)

x_coords = new_corners[:,0,0]
y_coords = new_corners[:,0,1]

new_w = int(np.max(x_coords) - np.min(x_coords))
new_h = int(np.max(y_coords) - np.min(y_coords))

R[0,2] -= np.min(x_coords)
R[1,2] -= np.min(y_coords)

rotated = cv2.warpAffine(img_rgb, R, (new_w, new_h))


scaled = cv2.resize(img_rgb, (300, 300), interpolation=cv2.INTER_LINEAR)

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(translated)
plt.title("Translated")

plt.subplot(2,2,3)
plt.imshow(rotated)
plt.title("Rotated (No Crop)")

plt.subplot(2,2,4)
plt.imshow(scaled)
plt.title("Scaled")

plt.tight_layout()
plt.show()
