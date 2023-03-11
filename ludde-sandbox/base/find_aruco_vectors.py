from base.utils.ARHelper import *
from base.utils.Calibrator import *
from base.utils.Camera import *
import cv2
from cv2 import aruco


camera = Camera()
ah = ARHelper()
cb = Calibrator()
cb.load_calibration("calibrations/matrix_good")


def draw_vectors(img, marker_corners, marker_ids, matrix, distortion):

    marker_size = 0.05
    axis_length = 0.02
    if len(marker_corners) > 0:
        for i in range(0, len(marker_ids)):
            rotation_vec, translation_vec, markerPoints = aruco.estimatePoseSingleMarkers(
                corners=marker_corners[i],
                markerLength=0.05,
                cameraMatrix=matrix,
                distCoeffs=distortion
            )
            if rotation_vec is not None:
                print("Rotation: ", rotation_vec)
                # todo fix multiple markers
                # z - blue , y - green, x - red
                cv2.drawFrameAxes(
                    image=img,
                    cameraMatrix=matrix,
                    distCoeffs=distortion,
                    rvec=rotation_vec,
                    tvec=translation_vec,
                    length=axis_length)

    return img


while True:

    image = camera.fetch()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    image, corners, ids = ah.find_markers(image)
    if ids is not None:
        image = draw_vectors(
            img=image,
            marker_corners=corners,
            marker_ids=ids,
            matrix=cb.camera_matrix,
            distortion=cb.distortion_coefficients
        )

    cv2.imshow("Video", image)


camera.stop()
cv2.destroyAllWindows()

