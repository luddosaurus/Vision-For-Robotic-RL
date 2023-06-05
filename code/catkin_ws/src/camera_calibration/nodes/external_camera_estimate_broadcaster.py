#! /usr/bin/env python3.8

import tf2_ros
import rospy

from camera_calibration.utils.TFPublish import TFPublish
from camera_calibration.utils.SaveMe import SaveMe

import camera_calibration.params.transform_frame_names as tfn
import camera_calibration.params.calibration as cal


class ExternalCameraEstimateBroadcaster(object):
    def __init__(self):
        self.pub_tf_static = tf2_ros.StaticTransformBroadcaster()

    def broadcast(self, transform):
        TFPublish.publish_static_stamped_transform(publisher=self.pub_tf_static, parent_name=tfn.world,
                                                   child_name=tfn.camera_estimate,
                                                   transform_stamped=transform)


def main():
    rospy.init_node('external_camera_estimate_broadcaster_node')
    transform = SaveMe.load_transforms(cal.external_calibration_path_position + 'camera_estimate.json')[0]  # returns a list of transforms
    broadcaster = ExternalCameraEstimateBroadcaster()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()
        broadcaster.broadcast(transform)


if __name__ == '__main__':
    main()
