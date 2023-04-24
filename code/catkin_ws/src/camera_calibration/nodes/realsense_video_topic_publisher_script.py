#! /usr/bin/python3.8 

import cv2
import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy
from sensor_msgs.msg import Image

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080


def main():
    rospy.init_node('realsense_video_topic_publisher_node')
    pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
    rate = rospy.Rate(1)

    # print(
    #     "f - Publish continuously\ng - Don't publish continuously\ns - Publish single image while not publishing continuously\nq - Quit")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, 30)

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # Start streaming
    pipeline.start(config)
    cv_bridge = CvBridge()

    send_continuous = True

    while not rospy.is_shutdown():
        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays

                color_image = np.asanyarray(color_frame.get_data())

                # Show images
                try:
                    img_message = cv_bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                except CvBridgeError as e:
                    print(e)
                pub.publish(img_message)
                # cv2.imshow('Realsense_color', color_image)
                # if send_continuous:
                #     pub.publish(img_message)
                # # rate.sleep()
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('f'):
                #     print('Publishing continuously')
                #     send_continuous = True
                # elif key == ord('g'):
                #     print('Not publishing continuously')
                #     send_continuous = False
                # elif key == ord('s') and not send_continuous:
                #     pub.publish(img_message)
                # elif key == ord('q'):
                #     break
        finally:
            # Stop streaming
            cv2.destroyAllWindows()
            pipeline.stop()


if __name__ == '__main__':
    main()
