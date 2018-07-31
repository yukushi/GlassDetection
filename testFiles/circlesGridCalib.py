import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

#images = glob.glob('*.jpeg')
images = glob.glob('*.png')

row = 9
col = 6

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Chess board
    #ret, corners = cv2.findChessboardCorners(gray, (row,col),None)

    #Circle
    ret, corners = cv2.findCirclesGrid(gray,(row,col),None)
    print(fname,ret)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (row,col), corners2,ret)
        cv2.imshow('img',img)

        #param
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        #change param
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        print(roi)
        if roi != (0,0,0,0):
            dst = dst[y:y+h, x:x+w]
        cv2.imshow('calibresult',dst)

        cv2.waitKey(0)

cv2.destroyAllWindows()
