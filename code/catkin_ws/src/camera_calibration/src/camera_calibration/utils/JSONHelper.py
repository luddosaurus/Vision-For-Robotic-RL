import json
from camera_calibration.params.calibration import config_path
from datetime import datetime


class JSONHelper(object):

    @staticmethod
    def read_json(file):
        with open(f'{config_path}/{file}.json') as json_file:
            json_data = json.load(json_file)
        return json_data

    @staticmethod
    def get_internal_calibration_parameters(file):
        json_data = JSONHelper.read_json(file)
        camera_name = json_data['camera_name']
        factory_settings = json_data['factory_settings']

        board_name = json_data['board_name']
        image_topic = json_data['image_topic']
        return camera_name, factory_settings, board_name, image_topic

    @staticmethod
    def get_board_parameters(board_name):
        json_data = JSONHelper.read_json('boards')
        board_parameters = json_data[board_name]
        return board_parameters

    @staticmethod
    def save_intrinsics(camera_name, camera_matrix, distortion, image_shape):
        if JSONHelper.check_name(camera_name):
            camera_name = f'{camera_name} - {str(datetime.now())}'

        json_data = JSONHelper.read_json('cameras')
        with open(f'{config_path}/cameras.json', 'w') as json_file:
            json_data[camera_name] = {
                'camera_matrix': [[camera_matrix[0][0], camera_matrix[0][1], camera_matrix[0][2]],
                                  [camera_matrix[1][0], camera_matrix[1][1], camera_matrix[1][2]],
                                  [camera_matrix[2][0], camera_matrix[2][1], camera_matrix[2][2]]],
                'distortion': [value for value in distortion[0]],
                'resolution': [image_shape[1], image_shape[0]]
            }
            print(json_data)
            json.dump(json_data, json_file)

    @staticmethod
    def check_name(camera_name):
        json_data = JSONHelper.read_json('cameras')
        return camera_name in json_data.keys()
