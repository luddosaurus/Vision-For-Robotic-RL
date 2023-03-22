import cv2 as cv2
import numpy as np
from cv.utils.find_contours import *
from cv.utils.stack_images import *
from cv.utils.image_transform import *

built_in_cam = 0
external_cam = 1
cam = external_cam
frameWidth = 640
frameHeight = 480

imgWidth = 640
imgHeight = 480

cap = cv2.VideoCapture(cam)
cap.set(3, frameHeight)
cap.set(4, frameWidth)
cap.set(10, 150)


def reorder(points):
    points = points.reshape((4, 2))
    reshaped_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)

    reshaped_points[0] = points[np.argmin(add)]
    reshaped_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    reshaped_points[1] = points[np.argmin(diff)]
    reshaped_points[2] = points[np.argmax(diff)]
    return reshaped_points


while True:
    success, img = cap.read()
    imgContour = img.copy()
    img = cv2.resize(img, (imgWidth, imgHeight))
    imgProcessed = pre_processing(img)
    biggestBox = get_contours2(imgProcessed, imgContour, 2500)

    if biggestBox.size != 0:
        imgWarped = warp_flat(img, biggestBox, imgHeight, imgWidth)
        imageArray = ([imgContour, imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
    else:
        imageArray = ([imgContour, img])

    stackedImages = stack_images(0.6, imageArray)
    cv2.imshow("WorkFlow", stackedImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
