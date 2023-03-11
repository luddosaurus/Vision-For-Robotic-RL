#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv2
import numpy as np


class FilterImage(object):

    def __init__(self):

        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge_object = CvBridge()

    def filter_image(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(450,350))
        cv2.imshow('Origin', img)

        #Apply the horizontal sobel operator with a kernel size of 3
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        cv2.imshow('X',sobelx)

        #Apply the vertical sobel operator with a kernel size of 3
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        cv2.imshow('Y',sobely)
        cv2.waitKey(0)

    def filter_image_custom_kernel(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(450,350))
        cv2.imshow('Origin', img)

        #Here we define the sobel operators
        #This are no more than a numpy matrix
        kernel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        kernel_y = np.array([[-1,0,1],[-2,-0,2],[-1,0,1]])

        #This part is where the magic happens
        #We convolve the image read with the kernels defined
        x_conv = cv2.filter2D(img,-1,kernel_x)
        y_conv = cv2.filter2D(img,-1,kernel_y)

        cv2.imshow('X',x_conv)

        cv2.imshow('Y',y_conv)
        cv2.waitKey(1)

    def filter_image_canny_edge(self, image):
        img = image
        cv2.imshow('Origin', img)

        #The canny detector uses two parameters appart from the image:
        #The minimum and maximum intensity gradient
        minV = 30
        maxV = 100

        edges = cv2.Canny(img,minV,maxV)

        cv2.imshow('Canny Edge',edges)
        cv2.waitKey(1)


    def filter_image_morph(self, image):
        img = image

        #Define a kernel for the erosion
        kernel_a = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel_a,iterations = 1)

        #Define a kernel for the dilation
        kernel_b = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(img,kernel_b,iterations = 1)

        #Define a kernel for the opening
        kernel_c = np.ones((7,7),np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_c)

        #Define a kernel for the closing
        kernel_d = np.ones((7,7),np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_d)

        cv2.imshow('Original',img)
        cv2.imshow('Erosion',erosion)
        cv2.imshow('Dilation',dilation)
        cv2.imshow('Opening',opening)
        cv2.imshow('Closing',closing)


        cv2.waitKey(1)

    def camera_callback(self, data):
        try:
            # We select bgr8 because its the OpenCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(
                data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(450,350))
        #Apply the horizontal sobel operator with a kernel size of 3
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)

        #Apply the vertical sobel operator with a kernel size of 3
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

        # FIlter
        # self.filter_image_morph(cv_image)
        self.filter_image_morph(sobely)




def main():
    rospy.init_node('filter_node', anonymous=True)
    showing_image_object = FilterImage()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")



if __name__ == '__main__':
    main()
