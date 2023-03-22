from code.utils.stack_images import *

img = cv2.imread('res/lena.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgHor = np.hstack((img, img))
# imgVer = np.vstack((imgHor, imgHor))

imgStack = stack_images(0.5, ([img, imgGray, img], [img, img, img]))

cv2.imshow("Image", imgStack)
cv2.waitKey(0)
