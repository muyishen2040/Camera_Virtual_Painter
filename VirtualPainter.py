import cv2
import os

import numpy as np

import HandTrackingModule as htm
import pycaw

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

folderPath = "Header"
pathList = os.listdir(folderPath)
#print(pathList)

detector = htm.handDetector(maxHands=1, detectionCon=0.75)
#############

drawingMask = [1,0,0,0]
selectingMask = [1,1,0,0]
testingMask = [1,1,1,0]

brushThickness = 8
eraserThickness = 40

#############
overlayList = []
for path in pathList:
    img = cv2.imread(f'{folderPath}\{path}')
    img = cv2.resize(img,(1280, 150))
    overlayList.append(img)

# print(len(overlayList))

mode = 0
curColor = (0, 0, 0)
px, py = 0, 0
while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmsList = detector.findPosition(img, draw=False)
    fingerList = detector.fingerStatus()
    #print(fingerList)

    if len(lmsList)!=0:
        if fingerList[1:] == drawingMask:
            x1, y1 = lmsList[8][1], lmsList[8][2]
            cv2.circle(img, (x1,y1), 15, curColor, cv2.FILLED)
            if px==0 and py==0:
                px, py = x1, y1
            if(curColor==(0, 0, 0)):
                cv2.line(imgCanvas, (px, py), (x1, y1), curColor, eraserThickness)
            else:
                cv2.line(imgCanvas, (px, py), (x1, y1), curColor, brushThickness)
            px, py = x1, y1
        elif fingerList[1:] == selectingMask:
            px, py = 0, 0
            x1,y1 = lmsList[8][1], lmsList[8][2]
            x2,y2 = lmsList[12][1], lmsList[12][2]
            if y2<150:
                if x2 < 390 and x2 > 290:
                    curColor = (255, 0, 0)
                    mode = 1
                elif x2 < 640 and x2 > 540:
                    curColor = (0, 255, 0)
                    mode = 2
                elif x2<840 and x2>740:
                    curColor = (0, 0, 255)
                    mode = 3
                elif x2<1150 and x2>1000:
                    curColor = (0, 0, 0)
                    mode = 4
            cv2.rectangle(img, (x1, y1 + 25), (x2, y2 - 25), curColor, cv2.FILLED)
        #elif fingerList[1:] == testingMask:
        #    print(f'({lmsList[12][1]},{lmsList[12][2]})')
        else:
            px, py = 0, 0

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    ih, iw, ic = overlayList[mode].shape
    img[0:ih,0:iw] = overlayList[mode]
    # cv2.imshow("Canvas",imgCanvas)
    cv2.imshow("Image", img)
    cv2.waitKey(1)