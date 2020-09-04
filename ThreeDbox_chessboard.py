# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:56:26 2020

@author: 11037
"""

import numpy as np
import cv2
import glob

from imutils.video import VideoStream

"""
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img
"""
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

def drawvirtualbox():
    cap = cv2.VideoCapture(0)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob('chessboard_image/*.jpg')

    print("test")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(10)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        #  print("ret",ret)
        if ret == True:
            objpoints.append(objp)
            # print("objpoint:", objpoints[0])
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # print("corners2:",corners2)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)

            cv2.imwrite("find.jpg", img)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("info", mtx, dist)
    # mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    mtx, dist = mtx, dist

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    print("objpoint:", objp)

    axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    # axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    while True:
        ret, img = cap.read()
        #img=cv2.resize('img',(500,500))
        cv2.waitKey(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        print(ret, "result^^^^^^^^^^^^^^^^^^")
        cv2.imshow('img00', img)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print("corners2:", corners2)
            print(objp.shape, (corners2.reshape(-1, 1, 2)).shape)
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objectPoints=objp,
                                                          imagePoints=corners2,
                                                          cameraMatrix=mtx,
                                                          distCoeffs=dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img, corners2, imgpts)

            cv2.imshow("img00", img)
            cv2.waitKey(1)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                print("stop AR")
                break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("stop AR")
            break

    cv2.destroyAllWindows()

