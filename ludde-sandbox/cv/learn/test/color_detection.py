from cv.utils.stack_images import *


def empty(a):
    pass


path = "../res/lambo.png"
windowName = "TrackBars"

cv2.namedWindow(windowName)
cv2.resizeWindow(windowName, 640, 640)
cv2.createTrackbar("Hue Min", windowName, 0, 179, empty)
cv2.createTrackbar("Hue Max", windowName, 19, 179, empty)
cv2.createTrackbar("Sat Min", windowName, 110, 255, empty)
cv2.createTrackbar("Sat Max", windowName, 240, 255, empty)
cv2.createTrackbar("Val Min", windowName, 153, 255, empty)
cv2.createTrackbar("Val Max", windowName, 255, 255, empty)

img = cv2.imread(path)
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    h_min = cv2.getTrackbarPos("Hue Min", windowName)
    h_max = cv2.getTrackbarPos("Hue Max", windowName)
    s_min = cv2.getTrackbarPos("Sat Min", windowName)
    s_max = cv2.getTrackbarPos("Sat Max", windowName)
    v_min = cv2.getTrackbarPos("Val Min", windowName)
    v_max = cv2.getTrackbarPos("Val Max", windowName)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)

    imgResult = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Image", stack_images(1, ([img, imgResult], [imgHSV, mask])))
    cv2.waitKey(1)
