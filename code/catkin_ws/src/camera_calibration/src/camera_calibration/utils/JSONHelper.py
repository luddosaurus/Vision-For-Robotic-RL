import json
from camera_calibration.params.calibration import config_path, extrinsic_calibration_results_path
from datetime import datetime
import os
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
import rospy


class JSONHelper(object):

    @staticmethod
    def read_json(file):
        with open(f'{file}.json') as json_file:
            json_data = json.load(json_file)
        return json_data

    @staticmethod
    def get_internal_calibration_parameters(file):
        json_data = JSONHelper.read_json(f'{config_path}/{file}')
        camera_name = json_data['camera_name']
        factory_settings = json_data['factory_settings']

        board_name = json_data['board_name']
        image_topic = json_data['image_topic']
        return camera_name, factory_settings, board_name, image_topic

    @staticmethod
    def get_board_parameters(board_name):
        json_data = JSONHelper.read_json(f'{config_path}/boards')
        board_parameters = json_data[board_name]
        return board_parameters

    @staticmethod
    def save_intrinsics(camera_name, camera_matrix, distortion, image_shape):
        if JSONHelper.check_name(camera_name):
            camera_name = f'{camera_name} - {str(datetime.now())}'

        json_data = JSONHelper.read_json(f'{config_path}/cameras')
        with open(f'{config_path}/cameras.json', 'w') as json_file:
            json_data[camera_name] = {
                'camera_matrix': [[camera_matrix[0][0], camera_matrix[0][1], camera_matrix[0][2]],
                                  [camera_matrix[1][0], camera_matrix[1][1], camera_matrix[1][2]],
                                  [camera_matrix[2][0], camera_matrix[2][1], camera_matrix[2][2]]],
                'distortion': [value for value in distortion],
                'resolution': [image_shape[1], image_shape[0]]
            }
            json.dump(json_data, json_file)

    @staticmethod
    def check_name(camera_name):
        json_data = JSONHelper.read_json(f'{config_path}/cameras')
        return camera_name in json_data.keys()

    @staticmethod
    def get_extrinsic_calibration_parameters(json_file):
        json_data = JSONHelper.read_json(f'{config_path}/{json_file}')
        board_name = json_data['board_name']
        camera_name = json_data['camera_name']
        mode = json_data['mode']
        camera_topic = json_data['camera_topic']
        memory_size = json_data['memory_size']
        load_data_directory = json_data['load_data_directory']
        save_data_directory = json_data['save_data_directory']
        return board_name, camera_name, mode, camera_topic, memory_size, load_data_directory, save_data_directory

    @staticmethod
    def get_camera_intrinsics(camera_name):
        json_data = JSONHelper.read_json(f'{config_path}/cameras')
        camera_data = json_data[camera_name]
        return camera_data

    @staticmethod
    def load_extrinsic_data(load_data_directory, eye_in_hand):

        if eye_in_hand:
            path = extrinsic_calibration_results_path + 'eye_in_hand/' + load_data_directory
        else:
            path = extrinsic_calibration_results_path + 'eye_to_hand/' + load_data_directory

        camera2target_data = JSONHelper.read_json(path + '/camera2target')
        hand2world_data = JSONHelper.read_json(path + '/hand2world')
        return JSONHelper.load_transform_list(camera2target_data), JSONHelper.load_transform_list(hand2world_data)

    @staticmethod
    def save_extrinsic_data(eye_in_hand, camera2target, hand2world, estimates, directory_name):
        time = str(datetime.now())
        if eye_in_hand:
            path = extrinsic_calibration_results_path + 'eye_in_hand/' + directory_name
        else:
            path = extrinsic_calibration_results_path + 'eye_to_hand/' + directory_name
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            path = path + time
            os.mkdir(path)
        JSONHelper.save_transform_list(camera2target, path + '/camera2target.json')
        JSONHelper.save_transform_list(hand2world, path + '/hand2world.json')
        JSONHelper.save_estimates(estimates, path + '/estimates.json')

    @staticmethod
    def load_transform_list(data):

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
    def save_transform_list(transforms, path):
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
    def save_estimates(estimates, path):
        methods = ['TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS']
        estimates_json = {}
        for (method, entry) in zip(methods, estimates):
            estimates_json[method] = {
                'time': entry.header.stamp.to_sec(),
                'frame_id': entry.header.frame_id,
                'child_frame_id': entry.child_frame_id,
                'translation': {
                    'x': entry.transform.translation.x,
                    'y': entry.transform.translation.y,
                    'z': entry.transform.translation.z
                },
                'rotation': {
                    'x': entry.transform.rotation.x,
                    'y': entry.transform.rotation.y,
                    'z': entry.transform.rotation.z,
                    'w': entry.transform.rotation.w
                }
            }
        with open(path, 'w') as f:
            json.dump(estimates_json, f)
