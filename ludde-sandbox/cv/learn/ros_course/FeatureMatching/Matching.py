#!/usr/bin/env python

import cv2
import numpy as np

image = cv2.imread('/home/user/catkin_ws_old/src/opencv_for_robotics_images/Unit_4/Course_images/test_e.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

keypoints = fast.detect(gray, None)
brief_keypoints, descriptor = brief.compute(gray, keypoints)

brief = np.copy(image)
non_brief = np.copy(image)

# Draw keypoints on top of the input image
cv2.drawKeypoints(image, brief_keypoints, brief, color=(0,250,250))
cv2.drawKeypoints(image, keypoints, non_brief, color=(0,35,250))

cv2.imshow('Fast corner detection',non_brief)
cv2.imshow('BRIEF descriptors',brief)

cv2.waitKey(0)
cv2.destroyAllWindows()