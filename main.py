# -*- coding: utf-8 -*-


import numpy as np
import cv2
from texttable import Texttable



#InputVideoNo
IN = 0

#HSV-Range
LO = np.array([40,70,150])
UP = np.array([85,255,255])

def labeling(frame,mask):
    """
    ラベリングを行う．
    数，高さと幅，面積，センター座標を取得する．
    必要外のラベルは排除する．
    """

    la, la2, la3, la4 = cv2.connectedComponentsWithStats(mask)

    #次元調整
    data = np.delete(la3, 0, 0)
    center = np.delete(la4, 0, 0)

    #テーブル整形
    table = Texttable()

    #ラベル数0の時に表示
    if(la-1 == 0):
        print("Not detected!")
        cv2.imshow("original",frame)

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
            print("skip")
            cv2.imshow("original",frame)
            break

        #テーブル作成
        table.add_rows([
        ["labelNum","width","height","centerX","centerY","size"],
        [i,w,h,centerX,centerY,data[i][4] ],
        ])

        #labelを囲うレクタングルプロット
        #見やすくするため枠を少し大きくとる
        sx2 = int((sx + w) + 20)
        sy2 = int((sy + h) + 20)
        sx = int(sx - 20)
        sy = int(sy - 20)

        #col = cv2.rectangle(col,(int(ax1),int(ay1)),(int(ax2),int(ay2)),(0,255,255),2)
        frame = cv2.rectangle(frame,(sx,sy),(sx2,sy2),(0,255,255),2)
        
    #テーブル表示
    print(table.draw())
    return frame

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

    while(True):
        ret,frame = cap.read()

        frame = cv2.medianBlur(frame,5)

        #HSV GreenMask
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv,LO,UP)

        #Gauusian for Hough
        enen = cv2.GaussianBlur(mask,(33,33),1)
        col = cv2.bitwise_and(frame,frame,mask=mask)

        #labeling
        frame = labeling(frame,mask)

        #検出範囲のプロット
        frame = cv2.rectangle(frame,(260,10),(400,470),(0,0,255),1)

        cv2.imshow("original",frame)
        #cv2.imshow("mask1",mask)
        cv2.imshow("color",col)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()
