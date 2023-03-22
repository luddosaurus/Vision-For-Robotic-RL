import numpy as np
import cv2
from cv2 import aruco as aruco
from cv.utils.image_transform import *
from cv.utils.ARHelper import *

internal_camera = 0
external_camera = 1

cap = cv2.VideoCapture(external_camera)

# Camera Size
camera_width = 640
camera_height = 480
cap.set(3, camera_width)
cap.set(4, camera_height)
cap.set(10, 100)

# ArUco Marker Settings
ah = ARHelper()

while True:
    # Run Camera
    success, image = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    image, corners, ids = ah.find_markers(image, debug=False)
    if ids is not None:
        first_corners = corners[0]
        warp_image = warp_flat(image, first_corners, camera_height, camera_width)
        cv2.imshow("Warp", warp_image)

    cv2.imshow("Video", image)


cap.release()
cv2.destroyAllWindows()


