#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class LoadVideo(object):

    def __init__(self):
        self.ctrl_c = False

        self.bridge_object = CvBridge()

    def shutdownhook(self):
        # works better than the rospy.is_shutdown()
        self.ctrl_c = True

    def video_detection(self, ):
        cap = cv2.VideoCapture("/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/chris.mp4")
        face_cascade = cv2.CascadeClassifier('/home/user/catkin_ws/src/unit3_exercises/haar_cascades/frontalface.xml')
        eyes_cascade = cv2.CascadeClassifier('/home/user/catkin_ws/src/unit3_exercises/haar_cascades/eye.xml')
        ScaleFactor = 1.2

        minNeighbors = 3
        while not self.ctrl_c:
            ret, frame = cap.read()

            img_original = cv2.resize(frame, (300, 200))
            img = cv2.resize(frame, (300, 200))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, ScaleFactor, minNeighbors)

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi = img[y:y + h, x:x + w]

                eyes = eyes_cascade.detectMultiScale(roi)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('Face_original', img_original)

            cv2.imshow('Face', img)

            cv2.waitKey(0)
        cap.release()


if __name__ == '__main__':
    rospy.init_node('load_video_node', anonymous=True)
    load_video_object = LoadVideo()
    try:
        load_video_object.video_detection()
        rospy.oncespin()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()