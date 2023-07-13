import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils

img_dir = '/home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/misc/assets'

img = cv.imread(img_dir + '/legos_red.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(markers):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[markers == label] = 255
    # detect contours in the mask and grab the largest one
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    # draw a circle enclosing the object
    ((x, y), r) = cv.minEnclosingCircle(c)
    cv.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# show_image = cv.resize(markers, None, fx=0.2, fy=0.2)
show_image_original = cv.resize(img, None, fx=0.2, fy=0.2)
# stacked_image = np.hstack((show_image_original, show_image))
cv.imshow('image', show_image_original)

cv.waitKey(0)
