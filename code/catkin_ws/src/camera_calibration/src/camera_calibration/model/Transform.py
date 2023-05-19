class Transform(object):
    def __init__(
            self,
            translation,
            rotation,
            frame_name="",
            parent_frame="",
            stamp=None,
            category=""
    ):
        # Essential
        self.translation_x, self.translation_y, self.translation_z = translation
        self.rotation_x, self.rotation_y, self.rotation_z, self.rotation_w = rotation
        # Extra

    def get_name(self):
        return f'aruco_[{self.aruco_id}]'

# todo 4x4 matrix, vectors , rotation matrix, panda frame
