import cv2
import numpy as np


src = "last.png"
img = cv2.imread(src)

h, w = img.shape[:2]
print(h,w)

x = 490*2
y = 350*2

zzz = 100

#FullScreen
pt1 = np.float32([[264,271],[375,272],[252,337],[391,338]])
pt2 = np.float32([[264,271],[375,272],[264,337],[375,338]])

H = cv2.getPerspectiveTransform(pt1,pt2)

dst = cv2.warpPerspective(img,H,(w,h))

re = cv2.resize(dst,(w*2,h*2))

print(H)

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
lo = np.array([0,40,100])
up = np.array([179,100,200])
mask = cv2.inRange(hsv,lo,up)


cv2.imshow("img",img)
cv2.imshow("img2",dst)
cv2.imshow("resizeImg",re)
cv2.waitKey(0)
cv2.destroyAllWindows()
