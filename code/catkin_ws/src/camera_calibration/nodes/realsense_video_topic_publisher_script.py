#! /usr/bin/python3.8

import cv2
import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy
from sensor_msgs.msg import Image

d435 = True

CAMERA_WIDTH = 1920 if d435 else 1280
CAMERA_HEIGHT = 1080 if d435 else 800

MAX_WIDTH_DEPTH = 1280
MAX_HEIGHT_DEPTH = 720


def main():
    rospy.init_node('realsense_video_topic_publisher_node')
    pub_color = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
    pub_aligned_depth = rospy.Publisher('/camera/aligned/image_raw', Image, queue_size=10)
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, min(CAMERA_WIDTH, MAX_WIDTH_DEPTH),
    #                      min(CAMERA_HEIGHT, MAX_HEIGHT_DEPTH), rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, min(CAMERA_WIDTH, MAX_WIDTH_DEPTH),
                         min(CAMERA_HEIGHT, MAX_HEIGHT_DEPTH),
                         rs.format.z16, 30)

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
    profile = pipeline.start(config)



    # Skip some frames for auto exposure

    for x in range(5):
        pipeline.wait_for_frames()


    cv_bridge = CvBridge()

    send_continuous = True

    while not rospy.is_shutdown():
        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                colorizer = rs.colorizer()

                # Convert images to numpy arrays

                color_image = np.asanyarray(color_frame.get_data())

                align = rs.align(align_to=rs.stream.color)
                frames = align.process(frames)

                aligned_depth_frame = frames.get_depth_frame()
                colorized_depth = colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

                images = np.hstack((color_image, colorized_depth))

                # Show images
                try:
                    img_message_color = cv_bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                    img_message_aligned_depth = cv_bridge.cv2_to_imgmsg(colorized_depth, encoding="bgr8")
                except CvBridgeError as e:
                    print(e)
                pub_color.publish(img_message_color)
                pub_aligned_depth.publish(img_message_aligned_depth)

                if CAMERA_HEIGHT >= 720:
                    images = cv2.resize(images, None, fx=0.5, fy=0.5)
                cv2.imshow('Realsense_color', images)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    rospy.Rate(1).sleep()
                    pipeline.stop()
                    del pipeline
                    rospy.signal_shutdown('Image view dismissed.')
                    break
                # rospy.Rate(100).sleep()
        finally:
            # Stop streaming
            cv2.destroyAllWindows()
            # pipeline.stop()
            # rospy.signal_shutdown('Image view dismissed.')


if __name__ == '__main__':
    main()
