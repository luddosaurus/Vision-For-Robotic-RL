import pyrealsense2 as rs

COLOR_WIDTH = 1280
COLOR_HEIGHT = 800


class ExtractParameters(object):
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.wrapper = rs.pipeline_wrapper(self.pipeline)
        self.wrapper_profile = self.config.resolve(self.wrapper)

        self.device = self.wrapper_profile.get_device()

        self.product_line = str(self.device.get_info(rs.camera_info.product_line))

        self.config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(self.config)

    def print_parameters(self):
        print("LOOK AT ME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for s in self.profile.get_streams():
            if s.stream_name() == 'Color':
                v = s.as_video_stream_profile()
                print(v.get_intrinsics())


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    wrapper = rs.pipeline_wrapper(pipeline)
    wrapper_profile = config.resolve(wrapper)

    device = wrapper_profile.get_device()

    product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    for s in profile.get_streams():
        if s.stream_name() == 'Color':
            v = s.as_video_stream_profile()
            print(v.get_intrinsics())


if __name__ == '__main__':
    main()
