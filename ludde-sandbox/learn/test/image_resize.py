import cv2
import numpy as np

img = cv2.imread('../res/lambo.png')


def print_dim(image):
    print("height = ", str(image.shape[0]))
    print("width = " + str(image.shape[1]))
    print("channels = " + str(image.shape[2]))


print_dim(img)

# Resize
imgResize = cv2.resize(img, (1000, 500))
print_dim(imgResize)

# Crop
imgCropped = img[0:200, 200:500]

cv2.imshow("Image", imgCropped)
cv2.waitKey(0)
