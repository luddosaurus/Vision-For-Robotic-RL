import cv2 as cv2
import numpy as np

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
    # print(f'points: {points}')
    add = points.sum(1)
    # print(f'add: {add}')
    reshaped_points[0] = points[np.argmin(add)]
    print(f'add: {add}')
    print(f'{reshaped_points[0]}')
    reshaped_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    # print(f'diff: {diff}')
    reshaped_points[1] = points[np.argmin(diff)]
    reshaped_points[2] = points[np.argmax(diff)]
    return reshaped_points