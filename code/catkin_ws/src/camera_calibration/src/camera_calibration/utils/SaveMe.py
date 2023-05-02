import json
import rospy
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
import os

from camera_calibration.params import calibration


class SaveMe:

    @staticmethod
    def save_transforms(transforms, path):
        data = []
        for transform in transforms:
            data.append({
                'time': transform.header.stamp.to_sec(),
                'frame_id': transform.header.frame_id,
                'child_frame_id': transform.child_frame_id,
                'translation': {
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z
                },
                'rotation': {
                    'x': transform.transform.rotation.x,
                    'y': transform.transform.rotation.y,
                    'z': transform.transform.rotation.z,
                    'w': transform.transform.rotation.w
                }
            })

        with open(path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load_transforms(filename):
        with open(calibration.external_calibration_path + filename, 'r') as f:
            data = json.load(f)

        transforms = []
        for d in data:
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.from_sec(d['time'])
            transform.header.frame_id = d['frame_id']
            transform.child_frame_id = d['child_frame_id']
            transform.transform.translation = Vector3(d['translation']['x'],
                                                      d['translation']['y'],
                                                      d['translation']['z'])
            transform.transform.rotation = Quaternion(d['rotation']['x'],
                                                      d['rotation']['y'],
                                                      d['rotation']['z'],
                                                      d['rotation']['w'])
            transforms.append(transform)

        return transforms

    @staticmethod
    def save_transform(transform, path):

        data = {
            'time': transform.header.stamp.to_sec(),
            'frame_id': transform.header.frame_id,
            'child_frame_id': transform.child_frame_id,
            'translation': {
                'x': transform.transform.translation.x,
                'y': transform.transform.translation.y,
                'z': transform.transform.translation.z
            },
            'rotation': {
                'x': transform.transform.rotation.x,
                'y': transform.transform.rotation.y,
                'z': transform.transform.rotation.z,
                'w': transform.transform.rotation.w
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f)
