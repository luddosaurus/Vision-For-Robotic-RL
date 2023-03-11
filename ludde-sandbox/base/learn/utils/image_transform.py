import cv2 as cv2
import numpy as np


def pre_processing(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(src=imgGray, ksize=(5, 5), sigmaX=1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel=kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    return imgThreshold


def get_warp(image, pixel_corners, img_height, img_width):
    ordered_box_points = reorder(pixel_corners)
    src_points = np.float32(ordered_box_points)
    dst_points = np.float32([
        [0, 0],
        [img_width, 0],
        [0, img_height],
        [img_width, img_height]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(image, matrix, (img_width, img_height))
    img_cropped = img_output[20:img_output.shape[0] - 20, 20:img_output.shape[1] - 20]
    img_cropped = cv2.resize(img_cropped, (img_width, img_height))
    return img_cropped


def reorder(points):
    points = points.reshape((4, 2))
    reshaped_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)

    reshaped_points[0] = points[np.argmin(add)]
    reshaped_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    reshaped_points[1] = points[np.argmin(diff)]
    reshaped_points[2] = points[np.argmax(diff)]
    return reshaped_points
