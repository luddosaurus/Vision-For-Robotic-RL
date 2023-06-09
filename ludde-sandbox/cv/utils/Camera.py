import cv2


class Camera:

    INTERNAL = 0
    EXTERNAL = 1
    camera_type = INTERNAL

    # Camera Size
    # camera_width = 640
    # camera_height = 480
    camera_width = 1485
    camera_height = 990
    frame_size = (camera_width, camera_height)

    # Capture
    cap = cv2.VideoCapture(camera_type)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    def fetch(self):
        _, img = self.cap.read()
        return img

    def stop(self):
        self.cap.release()
