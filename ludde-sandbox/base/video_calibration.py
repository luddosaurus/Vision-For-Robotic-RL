import cv2

from base.utils.Calibrator import Calibrator
from base.utils.Camera import Camera


camera = Camera()
cb = Calibrator()

# Run camera
while True:
    # Run Camera
    image = camera.fetch()
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    # Do Stuff
    chess_image = cb.find_and_save_chessboard_points(image)

    # Show Image
    cv2.imshow("Video", chess_image)

# Calibrate
calibrated_image = cb.calibrate_intrinsic(img=image)
cv2.imshow("Calibrated", calibrated_image)
cv2.waitKey(0)
print("Calibration pre-save \n", cb.camera_matrix)

# Save n Load calibration matrix
cb.save_calibration(save_name="test")

cb.find_error()

camera.stop()
cv2.destroyAllWindows()
