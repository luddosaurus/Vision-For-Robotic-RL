#! /usr/bin/python3.8

import cv2
import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy
from sensor_msgs.msg import Image


def get_parameters():
    color_height = rospy.get_param('realsense_video_topic_publisher_node/color_height')
    color_width = rospy.get_param('realsense_video_topic_publisher_node/color_width')
    depth_height = rospy.get_param('realsense_video_topic_publisher_node/depth_height')
    depth_width = rospy.get_param('realsense_video_topic_publisher_node/depth_width')
    align = rospy.get_param('realsense_video_topic_publisher_node/align')

    return color_height, color_width, depth_height, depth_width, align


def main():
    rospy.init_node('realsense_video_topic_publisher_node')

    color_height, color_width, depth_height, depth_width, align = get_parameters()

    # Topic publishers
    pub_color = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
    pub_depth = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=10)
    pub_aligned_depth = rospy.Publisher('/camera/aligned/image_raw', Image, queue_size=10)

    # Restart any connected realsense device, sleep allows it to boot up again
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
    rospy.Rate(0.5).sleep()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, 30)

    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)

    # Realsense boilerplate
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

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

                # Align depth and color frames
                align = rs.align(align_to=rs.stream.color)
                frames = align.process(frames)

                aligned_depth_frame = frames.get_depth_frame()
                aligned_colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
                colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                # images = np.hstack((color_image, colorized_depth))

                try:
                    img_message_color = cv_bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                    img_message_aligned_depth = cv_bridge.cv2_to_imgmsg(colorized_depth, encoding="bgr8")
                    img_message_depth = cv_bridge.cv2_to_imgmsg(colorized_depth, encoding="bgr8")
                except CvBridgeError as e:
                    print(e)
                pub_color.publish(img_message_color)
                pub_depth.publish(img_message_depth)
                if align:
                    pub_aligned_depth.publish(img_message_aligned_depth)


                # Show images
                # cv2.imshow('Realsense_color', images)
                #
                # key = cv2.waitKey(1) & 0xFF
                #
                # if key == ord('q'):
                #     cv2.destroyAllWindows()
                #     rospy.Rate(1).sleep()
                #     pipeline.stop()
                #     del pipeline
                #     rospy.signal_shutdown('Image view dismissed.')
                #     break
                # rospy.Rate(100).sleep()
        finally:
            # Stop streaming
            cv2.destroyAllWindows()
            pipeline.stop()
            # pipeline.stop()
            # rospy.signal_shutdown('Image view dismissed.')


if __name__ == '__main__':
    main()
