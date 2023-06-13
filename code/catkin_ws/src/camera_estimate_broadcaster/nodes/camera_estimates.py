#! /usr/bin/env python3.8
import tf
import tf2_ros
import rospy

from camera_calibration.utils.TFPublish import TFPublish
import os
import rospkg

from camera_calibration.utils.JSONHelper import JSONHelper
import camera_calibration.params.calibration as cal


class ExternalCameraEstimateBroadcaster(object):
    def __init__(self, camera_data):
        # self.pub_tf_static = tf2_ros.StaticTransformBroadcaster()
        self.transforms = camera_data
        self.publishers = {key: tf2_ros.StaticTransformBroadcaster() for key in self.transforms.keys()}

    def broadcast(self):
        for key in self.transforms.keys():
            frame_id = self.transforms[key]['frame_id']
            child_frame_id = self.transforms[key]['child_frame_id']
            translation = []
            rotation = []
            for coordinate_key in ['x', 'y', 'z', 'w']:
                if coordinate_key != 'w':
                    translation.append(self.transforms[key]['translation'][coordinate_key])
                rotation.append(self.transforms[key]['rotation'][coordinate_key])

            TFPublish.publish_transform(publisher=self.publishers[key], parent_name=frame_id, child_name=child_frame_id,
                                        translation=translation, rotation=rotation)
        # TFPublish.publish_static_stamped_transform(publisher=self.pub_tf_static, parent_name=tfn.world,
        #                                            child_name=tfn.camera_estimate,
        #                                            transform_stamped=transform)


def main():
    rospy.init_node('camera_estimates')

    cameras_file = rospy.get_param('camera_estimates/camera_transforms')
    cameras_path = os.path.join(rospkg.RosPack().get_path('camera_estimate_broadcaster'),
                                f'camera_transforms/{cameras_file}')
    data = JSONHelper.get_camera_estimates(cameras_path)

    broadcaster = ExternalCameraEstimateBroadcaster(camera_data=data)
    first_broadcast = True
    rate = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        if first_broadcast:
            for i in range(10):
                broadcaster.broadcast()
            first_broadcast = False
        broadcaster.broadcast()
        rate.sleep()


if __name__ == '__main__':
    main()
