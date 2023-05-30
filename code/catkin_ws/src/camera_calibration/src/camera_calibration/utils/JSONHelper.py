import json
from camera_calibration.params.calibration import config_path


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
        board = json_data['board']
        image_topic = json_data['image_topic']
        return camera_name, factory_settings, board, image_topic

    @staticmethod
    def get_board_parameters(board_name):
        json_data = JSONHelper.read_json('boards')
        board_parameters = json_data[board_name]
        return board_parameters
