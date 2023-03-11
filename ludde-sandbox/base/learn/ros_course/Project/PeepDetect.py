#!/usr/bin/env python

import numpy as np
import cv2


class PeepDetect:

    def __init__(self):
        pass

    def find_peep(self, image):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        img = image

        imX = 720
        imY = 500
        img = cv2.resize(img, (imX, imY))

        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            # Center in X
            medX = xB - xA
            xC = int(xA + (medX / 2))

            # Center in Y
            medY = yB - yA
            yC = int(yA + (medY / 2))

            # Draw a circle in the center of the box
            cv2.circle(img, (xC, yC), 1, (0, 255, 255), -1)

            # display the detected boxes in the original picture
            cv2.rectangle(img, (xA, yA), (xB, yB),
                          (255, 255, 0), 2)

        cv2.imshow('Peeps', img)
        # cv2.waitKey(1)
