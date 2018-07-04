# -*- coding: utf-8 -*-


import numpy as np
import cv2
from texttable import Texttable



#InputVideoNo
IN = 0

#HSV-Range
LO = np.array([40,70,150])
UP = np.array([85,255,255])

#Transformation matrix
H = [[-5.24959249e+01, -6.37688338e+01, 1.69839096e+04],
     [ 1.02243118e+00, -1.22654622e+02, 1.84377541e+04],
     [ 3.31747014e-03, -2.04791218e-01, 1.00000000e+00]]
H = np.array(H)

def pixelDistance(c):
    point1  = np.array(c[0])
    point2  = np.array(c[1])

    u = point2 - point1
    dist = np.linalg.norm(u)

    return point1,point2,dist

def labeling(frame,mask,jg=0):
    """
    ラベリングを行う． 数，高さと幅，面積，センター座標を取得する．
    必要外のラベルは排除する．
    """

    point = []
    la, la2, la3, la4 = cv2.connectedComponentsWithStats(mask)

    #次元調整
    data = np.delete(la3, 0, 0)
    center = np.delete(la4, 0, 0)

    #テーブル整形
    table = Texttable()

    #ラベル数0の時に表示
    if(la-1 == 0):
        print('\rNot Detected!',end='')
        return frame

    for i in range(la-1):
        #labelの始点とサイズ情報
        sx = data[i][0]
        sy = data[i][1] 
        w = data[i][2]
        h = data[i][3]

        #中心
        centerX = center[i][0]
        centerY = center[i][1]
        
        #範囲外無視，ラベル数制限
        if(data[i][4] <=1) or (la-1 >= 30) or (centerX<260) or (centerX>400):
            print('\rSkip!',end='')
            cv2.imshow("original",frame)
            return frame

        #labelを囲うレクタングルプロット
        #見やすくするため枠を少し大きくとる
        sx2 = int((sx + w) + 20)
        sy2 = int((sy + h) + 20)
        sx = int(sx - 20)
        sy = int(sy - 20)

        #テーブル作成
        table.add_rows([
        ["labelNum","width","height","centerX","centerY","size"],
        [i,w,h,centerX,centerY,data[i][4] ],
        ])

        point.append([centerX,centerY])

        frame = cv2.rectangle(frame,(sx,sy),(sx2,sy2),(0,255,255),2)
        
    if(la-1 == 2 and jg == 1 and point is not None):
        #テーブル表示
        print(table.draw())
        #距離を求める
        p1,p2,dist = pixelDistance(point)
        print("| Distance = ",dist)
        print("+----------+-------+--------+---------+---------+------+")
        #距離プロット
        frame = cv2.line(frame2,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(255,255,0),3)

        font = cv2.FONT_HERSHEY_PLAIN
        frame = cv2.putText(frame2,str(dist),(int(p1[0]+20),300),font,4,(0,0,255),2,cv2.LINE_AA)
    elif(la-1 == 2 and jg == 0 and point is not None and len(point) is not 0):
        p1,p2,dist = pixelDistance(point)
        frame = cv2.line(frame,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(255,0,0),3)
        
    return frame

def pTrans(frame):
    """
    射影変換を行う．
    事前に求めたホモグラフィ行列を用いる．
    """
    hframe = cv2.warpPerspective(frame,H,(frameW,frameH))
    #hframe = cv2.resize(hframe,(frameW*2,frameH*2))
    return hframe

def circleDetect():
    """
    ハフ変換を用いて円の検出を行う．
    """
    circles = cv2.HoughCircles(enen,cv2.HOUGH_GRADIENT,1,100,param1=30,param2=30,minRadius=20,maxRadius=0) 
    print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(frame,(i[0],i[1]),i[2],(255,255,0),2)

if __name__ == '__main__':
    cap  = cv2.VideoCapture(IN)

    while(cap.isOpened()):
        ret,frame = cap.read()
        ###global h,w
        frameH,frameW = frame.shape[:2]

        frame = cv2.medianBlur(frame,5)

        #HSV GreenMask
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,LO,UP)

        #Gauusian for Hough
        enen = cv2.GaussianBlur(mask,(33,33),1)
        green = cv2.bitwise_and(frame,frame,mask=mask)

        #Perspective Trans
        frame2 = pTrans(frame)
        frame2hsv = cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV) 
        syaMask = cv2.inRange(frame2hsv,LO,UP)

        #labeling
        frame = labeling(frame,mask)
        frame2 = labeling(frame2,syaMask,1)

        #検出範囲のプロット
        frame = cv2.rectangle(frame,(260,10),(400,470),(0,0,255),1)
        frame2 = cv2.rectangle(frame2,(260,10),(400,470),(0,0,255),1)


        cv2.imshow("original",frame)
        #cv2.imshow("GreenCheck",green)
        #cv2.imshow("pTrans",syaMask)
        cv2.imshow("pTransLabel",frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()
