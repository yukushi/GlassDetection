import numpy as np
import cv2



img = cv2.imread("labelTest.jpg")

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lo = np.array([50,100,100])
up = np.array([140,255,255])
mask = cv2.inRange(hsv,lo,up)
col = cv2.bitwise_and(img,img,mask=mask)

la, la2, la3, la4 = cv2.connectedComponentsWithStats(mask)

print(la,la2,la3,la4,sep='\n\n')
print('\n')

cv2.imshow("labelTest",mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
