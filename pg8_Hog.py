import cv2
hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img=cv2.imread(r"D:\4SF23CI052-DHWANI\IPCV\pedestrians.jpg")

(rects,weights)=hog.detectMultiScale(img,winStride=(4,4),padding=(8,8),scale=1.05)
for(x,y,w,h) in rects:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('HOG Pedestrian Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
