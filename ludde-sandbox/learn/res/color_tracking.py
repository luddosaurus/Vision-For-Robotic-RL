import cv2 as cv2
import numpy as np
from code.utils.find_contours import *

built_in_cam = 0
external_cam = 1
cam = built_in_cam
frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(cam)
cap.set(3, frameHeight)
cap.set(4, frameWidth)
cap.set(10, 150)

my_colors = [
    [48, 44, 74, 119, 255, 255],
    [49, 93, 45, 85, 176, 168]]
my_color_values = [[253, 165, 49], [63, 145, 22]]
points_to_draw = []  # [x, y, colorId]


def find_color(image, colors, image_result, color_value):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    count = 0
    new_points = []
    for color in colors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(img_hsv, lower, upper)
        x, y = get_contours(mask, image_result)
        cv2.circle(image_result, (x, y), 10, color_value[count], cv2.FILLED)
        if x != 0 and y !=0:
            new_points.append([x, y, count])

        count += 1
        # cv2.imshow(str(color[0]), mask)

    return new_points


def draw_on_canvas(points, color_values, image_result):
    for point in points:
        cv2.circle(
            image_result,
            (point[0], point[1]),
            10,
            color_values[point[2]],
            cv2.FILLED)


while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = find_color(img, my_colors, imgResult, my_color_values)
    if len(newPoints) != 0:
        for newP in newPoints:
            points_to_draw.append(newP)

    if len(points_to_draw) != 0:
        draw_on_canvas(points_to_draw, my_color_values, imgResult)

    # cv2.imshow("Cam", img)
    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
