import cv2
from PIL import Image
import numpy as np


def get_limits(color):
    c = np.uint8([[color]])  # here insert the bgr values which you want to convert to hsv
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    print(hsvC[0][0][0])
    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


colors = [  # [0, 255, 255],
    # [0, 0, 255],
    # [255, 0, 0],
    [0, 255, 0]
]

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    final_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for color in colors:

        lowerLimit, upperLimit = get_limits(color=color)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        final_mask = cv2.bitwise_or(final_mask, mask)
        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    masked_image = cv2.bitwise_and(frame, frame, mask=final_mask)
    stacked = np.hstack((frame, masked_image))

    cv2.imshow('frame', stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
