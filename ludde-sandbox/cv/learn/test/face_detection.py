import cv2 as cv2
import numpy as np

# Cascade model_old to find faces/cats/cars/etc
faceCascade = cv2.CascadeClassifier("res/haarcascades/haarcascade_frontalface_default.xml")
img = cv2.imread("../res/lena.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(imgGray, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Output", imgGray)
cv2.waitKey(0)
