import matplotlib.pyplot as plt
import cv2

img1=cv2.imread(r"D:\DHWANI\ENGINEERING\VI\IPCV\ipcv_lab\image.jpg")
img=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

height,width,channels=img.shape
mh=height//2
mw=width//2

tl=img[:mh,:mw]
tr=img[:mh,mw:width]
bl=img[mh:height,:mw]
br=img[mh:height,mw:width]

fig,axes=plt.subplots(2,2,figsize=(8,8))
axes[0,0].imshow(tl)
axes[0,0].set_title("Top Left")
axes[0,0].axis("on")

axes[0,1].imshow(tr)
axes[0,1].set_title("Top Right")
axes[0,1].axis("on")

axes[1,0].imshow(bl)
axes[1,0].set_title("Bottom Left")
axes[1,0].axis("on")

axes[1,1].imshow(br)
axes[1,1].set_title("Bottom RIght")
axes[1,1].axis("on")

plt.tight_layout()
plt.show()
