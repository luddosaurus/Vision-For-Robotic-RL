#! /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

if __name__ == '__main__':
    rospy.init_node('webcam_node')
    cap = cv2.VideoCapture(0)

    image_publisher = rospy.Publisher('webcam/color/image_raw', Image, queue_size=10)
    cv_bridge = CvBridge()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while not rospy.is_shutdown():
        try:
            while cap.isOpened():

                ret, img = cap.read()
                if ret:
                    try:
                        img_message = cv_bridge.cv2_to_imgmsg(img, encoding='bgr8')
                    except CvBridgeError as e:
                        print(e)
                    image_publisher.publish(img_message)

                    # cv2.imshow('Image', img)
                    #
                    # key = cv2.waitKey(1) & 0xFF
                    # if key == ord('q'):
                    #     break

        finally:
            cv2.destroyAllWindows()
            cap.release()
