import cv2

img_dir = '/home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/misc/assets'

im = cv2.imread(img_dir + '/legos_4.jpg')
im = cv2.rotate(im, rotateCode=2)
# assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
th2_1 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                              cv2.THRESH_BINARY, 11, 2)
th3_1 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                              cv2.THRESH_BINARY, 11, 2)

# Otsu's thresholding
ret2, th2 = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(imgray, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

test = cv2.inRange(im, (0, 0, 40), (60, 60, 255))

cv2.drawContours(im, contours, -1, (0, 255, 0), 3)

cv2.imshow('gray', cv2.resize(imgray, None, fx=0.3, fy=0.3))
cv2.imshow('image', cv2.resize(im, None, fx=0.3, fy=0.3))
cv2.imshow('test', cv2.resize(test, None, fx=0.3, fy=0.3))
cv2.waitKey(0)
