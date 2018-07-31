import numpy as np
import cv2

URL = 1
cap = cv2.VideoCapture(URL)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((4*3,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)

objpoints = []
imgpoints = []

row = 4
col = 3

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #ChessBoard
    #ret, corners = cv2.findChessboardCorners(gray, (row,col),None)

    #CircleGrid
    ret, corners = cv2.findCirclesGrid(gray,(row,col),None)
    print(ret)

    if(ret == True):
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(frame, (row,col), corners2,ret)
        cv2.imshow('frame',frame)

        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        #print(ret,mtx,dist,rvecs,tvecs)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
