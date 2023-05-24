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
        self.translation = translation
        self.rotation = rotation
        self.translation_x, self.translation_y, self.translation_z = translation
        self.rotation_x, self.rotation_y, self.rotation_z, self.rotation_w = rotation
        # Extra

    def transform_matrix(self):
        pass

    def rotation_matrix(self):
        pass

    def rotation(self):
        return self.rotation

    def translation(self):
        return self.translation

    def panda_frame(self):
        pass

