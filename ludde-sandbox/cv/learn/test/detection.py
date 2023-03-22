import cv2 as cv2
import numpy as np
from cv.utils.find_contours import *
from cv.utils.stack_images import *

built_in_cam = 0
external_cam = 1
cam = built_in_cam
frameWidth = 640
frameHeight = 480
faceCascade = cv2.CascadeClassifier("res/haarcascades/haarcascade_frontalface_default.xml")
minArea = 10000
color = (200, 50, 200)

imgWidth = 640
imgHeight = 480

cap = cv2.VideoCapture(cam)
cap.set(3, frameHeight)
cap.set(4, frameWidth)
cap.set(10, 150)

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in faces:
        area = w*h
        if area > minArea:
            cv2.rectangle(imgGray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imgGray, "Face",
                        (x, y-5),
                        color=color,
                        fontScale=2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow("WorkFlow", imgGray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
