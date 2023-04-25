#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy
from src.camera_calibration.utils.TFPublish import TFPublish
from src.camera_calibration.utils.TypeConverter import TypeConverter

import geometry_msgs
from std_msgs.msg import UInt8MultiArray
import tf
import tf2_ros
from tf.transformations import quaternion_matrix
import numpy as np
from time import time


from src.camera_calibration.params.attached_arucos import table_arucos, arm_arucos
from src.camera_calibration.utils.MeanHelper import MeanHelper


class StaticCameraPositionEstimator(object):
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
        self.marker_subscriber = rospy.Subscriber(
            '/detected_aruco_marker_ids', 
            UInt8MultiArray, 
            self.marker_callback
            )
        self.transformations = list()
        self.start_time = time()

    def marker_callback(self, message):
        # marker_ids = message.data
        marker_ids = [int(x) for x in message.data]

        rate = rospy.Rate(10.0)


        for aruco in marker_ids:

            source_frame = 'world'
            target_frame = f'aruco_to_camera_[{aruco}]'

            try:
                transform = self.tfBuffer.lookup_transform(
                    source_frame=source_frame, 
                    target_frame=target_frame, 
                    time=rospy.Time())
                
                self.transformations.append(transform)

            except (
                tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException):

                rate.sleep()

        if len(self.transformations) > 0:
            self.start_time = time()
            avg_transform = self.create_average_transform_stamped_message(self.transformations)
            TFPublish.publish_static_stamped_transform(
                publisher=self.pub_aruco_tf,
                transform_stamped=avg_transform,
                parent_name="world",
                child_name="camera_position"
            )
            
            if len(self.transformations) % 10:
                print(avg_transform)

            rospy.sleep(1)

            if len(self.transformations) > 500:
                print(avg_transform)
                self.transformations.pop(0)
            
        rate.sleep()


    def create_average_transform_stamped_message(self, transformations):
        
        mean_translation, mean_rotation = MeanHelper.riemannian_mean(transformations)

        # New 
        avg_transform = TypeConverter.vectors_to_stamped_transform(
            translation=mean_translation,
            rotation=mean_rotation,
            parent_frame=transformations[0].header.frame_id,
            child_frame=transformations[0].child_frame_id
        )

        return avg_transform


# Broadcast Transform: Base -> Camera
if __name__ == '__main__':
    rospy.init_node('external_camera_node')
    static_camera_position_estimator = StaticCameraPositionEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')

