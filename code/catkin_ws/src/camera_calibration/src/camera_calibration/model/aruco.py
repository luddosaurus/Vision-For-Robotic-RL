#! /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python

# ArUco container with ID, and offset
class ArUco(object):
    def __init__(self, aruco_id, translation, rotation):
        self.aruco_id = aruco_id
        self.translation = translation
        self.rotation = rotation
        self.translation_x, self.translation_y, self.translation_z = translation
        self.rotation_x, self.rotation_y, self.rotation_z, self.rotation_w = rotation

    def get_name(self):
        return f'aruco_[{self.aruco_id}]'
