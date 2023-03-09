#!/usr/bin/env python
import numpy as np
from skimage import exposure
from skimage import feature
import cv2

img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/test_e.jpg')

imX = 720
imY = 1080
img = cv2.resize(img, (imX, imY))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                            visualize=True)

hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

cv2.imshow('features', hogImage)
cv2.waitKey(0)
cv2.destroyAllWindows()