import cv2 as cv2
from cv.utils.stack_images import *


def get_contours(image, draw_image=None, target_size=500):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > target_size:
            if draw_image is not None:
                cv2.drawContours(draw_image, cnt, -1, (255, 0, 0), 3)

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)

    return x + w // 2, y


def get_contours_card(image, draw_image=None, target_size=500):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_box = np.array([])
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > target_size:
            if draw_image is not None:
                cv2.drawContours(draw_image, cnt, -1, (255, 0, 0), 3)

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest_box = approx
                max_area = area

    return biggest_box


def get_contours2(img, draw_image, target_size=5000):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > target_size:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(draw_image, biggest, -1, (255, 0, 0), 20)
    return biggest
