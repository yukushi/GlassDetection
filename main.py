# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
from mouseEvent import *
from texttable import Texttable
import math
import subprocess



#InputVideoNo
IN = 1

#HSV-Range
LO = np.array([40,70,150])
UP = np.array([85,255,255])

H = [[ 3.38755978e+01,  4.28291367e+01, -1.03256465e+04],
      [-1.90794828e-01,  9.08031213e+01, -1.37086084e+04],
      [-7.85163902e-04,  1.37403683e-01,  1.00000000e+00]]
H = np.array(H)

PIX_DIST = 3.835013386

#Degree and Height
DEG = 45        #レーザポインタの照射角度
HEIGHT = 720    #レーザポインタの高さ

point_position = 0

expoValue = "100"

def pixelDistance(c):
    point1  = np.array(c[0])
    point2  = np.array(c[1])

    u = point2 - point1
    dist = np.linalg.norm(u)
    dist = dist*3.875013386

    return point1,point2,dist

def labeling(frame,mask,jg=0):
    """
    ラベリングを行う．
     数，高さと幅，面積，センター座標を取得する．
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
        print('\rNot Detected!                 ',end='')
        return frame
    elif(la-1 >= 3):
        print('\rToo many objects for labeling',end='')
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
        if(data[i][4] <=2) or (la-1 >= 30) or (centerX<260) or (centerX>400) or (centerY<100):
            print('\rSkip!                        ',end='')
            #cv2.imshow("original",frame)
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

        frame = cv2.rectangle(frame,(sx,sy),(sx2,sy2),(0,255,255),1)
        
    if(la-1 == 2 and jg == 1 and point is not None):
        #テーブル表示
        print("\n")
        print(table.draw())

        #2点間の距離を求める
        p1,p2,dist = pixelDistance(point)
        print("| Distance = ",dist)
        print("+----------+-------+--------+---------+---------+------+")

        #2点間を結ぶラインをプロット
        frame = cv2.line(frame2,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(255,220,0),3)

        #ガラス面推測
        gLine = int((p1[1]+p2[1])/2)
        frame = cv2.line(frame2,(0,gLine),(640,gLine),(0,0,255),3)

        font = cv2.FONT_HERSHEY_PLAIN
        #距離の表示
        dist_int = dist
        dist = str(int(dist))+'mm'

        #照射点までの距離
        laser_irradiation_point = int(HEIGHT * math.tan(math.radians(DEG)))

        for i in range(2):
            if(center[i][0] > 270 and center[i][0] < 310 and center[i][1] > 320 and center[i][1] < 360):
                frame = cv2.putText(frame2,"Point",(int(center[i][0]+40),int(center[i][1])),font,1,(55,255,55),1,cv2.LINE_AA)
                point_position = i
                print(point_position) # 0->ガラスに反射した場合,1->ガラスの手前に照射した場合
                if(point_position == 1):
                    glass_dist = laser_irradiation_point + int(dist_int)/2
                else:
                    glass_dist = laser_irradiation_point - int(dist_int)/2
                print("glass_dist=",glass_dist)
                frame = cv2.putText(frame2,str(glass_dist)+"mm",(80,gLine+50),font,2,(255,55,0),2,cv2.LINE_AA) #ガラスまでの距離プロット
            else:
                frame = cv2.putText(frame2,"Reflection Point",(int(center[i][0]+40),int(center[i][1])),font,1,(55,255,55),1,cv2.LINE_AA)
        
        frame = cv2.arrowedLine(frame2,(30,frameH),(30,gLine),(255,55,0),thickness=4) #矢印
        
        #2点間の距離プロット
        #frame = cv2.putText(frame2,dist,(int(p1[0]+20),300),font,2,(255,255,0),2,cv2.LINE_AA)
        #Glassとプロット
        frame = cv2.putText(frame2,"Glass",(0,gLine),font,2,(0,0,255),1,cv2.LINE_AA)
        
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

def changeExposureValue():
    """
    カメラの露出調整を1~5キーを使って行う．
    """
    global expoValue

    if cv2.waitKey(1) & 0xFF == ord('1'):
        expoValue= "150"
    elif cv2.waitKey(1) & 0xFF == ord('2'):
        expoValue = "100"
    elif cv2.waitKey(1) & 0xFF == ord('3'):
        expoValue = "50"
    elif cv2.waitKey(1) & 0xFF == ord('4'):
        expoValue = "25"
    elif cv2.waitKey(1) & 0xFF == ord('5'):
        expoValue = "10"

    #露出調整
    cmd = 'v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_absolute={}'
    cmd = cmd.format(expoValue)
    ret = subprocess.check_output(cmd,shell=True)

if __name__ == '__main__':
    cap  = cv2.VideoCapture(IN)

    while(cap.isOpened()):
        ret,frame = cap.read()
        frameH,frameW = frame.shape[:2]
        frame = cv2.medianBlur(frame,5)

        #Mouse event
        #cv2.setMouseCallback("original",clickPoint,frame)

        #ガンマ補正
        gamma = 0.70
        gamma_cvt = np.zeros((256,1),dtype = 'uint8')
        for aaa in range(256):
            gamma_cvt[aaa][0] = 255*(float(aaa)/255) ** (1.0/gamma)
        frame = cv2.LUT(frame,gamma_cvt)

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
        #frame = labeling(frame,mask)
        frame2 = labeling(frame2,syaMask,1)

        #検出範囲のプロット
        #frame = cv2.rectangle(frame,(260,10),(400,470),(0,0,255),1)
        frame2 = cv2.rectangle(frame2,(260,10),(400,470),(0,0,255),1)
        #照射予測点
        frame2 = cv2.rectangle(frame2,(270,320),(310,360),(0,0,255),1)
        
        #クリックで指定した変換結果を表示
        #syaeiFrame(frame)
        #frame = clickPointCircle(frame) #クリック点プロット

        #露出度調整
        changeExposureValue()

        #cv2.imshow("original",frame)
        cv2.imshow("mask",mask)
        cv2.imshow("pTransLabel",frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()
