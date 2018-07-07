import cv2
import numpy as np

cPoint = []
H2 = []

def clickPoint(event,x,y,flags,param):
    global cPoint
    global H2

    if event == cv2.EVENT_LBUTTONUP:
        cPoint.append([x,y])
        print('\rPointを設定 : %s' % cPoint,end='')

        if len(cPoint) == 4:
            pt1 = np.float32(cPoint)
            pt2 = np.float32([cPoint[0],cPoint[1],[cPoint[0][0],cPoint[2][1]],[cPoint[1][0],cPoint[3][1]]])

            H2 = cv2.getPerspectiveTransform(pt1,pt2)
            print("\n%s\n"%H2)
            dst = cv2.warpPerspective(param,H2,(640,480))

            print("変換結果表示\n")
            cPoint = []

def syaeiFrame(frame):
    if len(H2) == 3:
        dst = cv2.warpPerspective(frame,H2,(640,480))
        cv2.imshow("pTrans999",dst)
    else:
        pass
