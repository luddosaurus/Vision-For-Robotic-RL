from code.utils.stack_images import *


def get_contours(image, image_contour):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(image_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            obj_corner = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            object_type = "None"
            if obj_corner == 3:
                object_type = "Triangle"
            elif obj_corner == 4:
                asp_ratio = w / float(h)
                if 0.95 < asp_ratio < 1.05:
                    object_type = "Square"
                else:
                    object_type = "Rectangle"
            elif obj_corner > 4:
                object_type = "Circle"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (200, 0, 200), 2)
            cv2.putText(imgContour, object_type,
                        (x + (w // 2) - 10, y + (h // 2) - 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (50, 0, 50), 2)


path = "../res/shapes.png"
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)

get_contours(imgCanny, imgContour)

imgBlank = np.zeros_like(img)
imgStack = stack_images(1 / 2, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgBlank]))

cv2.imshow("Image", imgStack)
cv2.waitKey(0)
