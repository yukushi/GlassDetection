import numpy as np
import cv2


cPoint = []

def mouseTest(event,x,y,flags,param):

    global cPoint

    if event == cv2.EVENT_LBUTTONUP:
        cPoint.append([x,y])
        print('\rPointを設定 : %s' % cPoint,end='')

        if len(cPoint) == 4:
            pt1 = np.float32(cPoint)
            pt2 = np.float32([cPoint[0],cPoint[1],[cPoint[0][0],cPoint[2][1]],[cPoint[1][0],cPoint[3][1]]])

            H = cv2.getPerspectiveTransform(pt1,pt2)
            print("\n%s\n"%H)
            dst = cv2.warpPerspective(img,H,(640,480))

            print("変換結果表示\n")
            cv2.imshow("henkan",dst)

            cPoint = []

img = cv2.imread("last.png",1)
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("img",mouseTest)

while(True):
    cv2.imshow("img",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
