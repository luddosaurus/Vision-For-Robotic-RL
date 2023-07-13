import cv2
import numpy as np

img_dir = '/home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/misc/assets'
img = cv2.imread(img_dir + '/legos_4.jpg')
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

show_image = cv2.resize(res2, None, fx=0.2, fy=0.2)
show_image_original = cv2.resize(img, None, fx=0.2, fy=0.2)
stacked_image = np.hstack((show_image_original, show_image))
cv2.imshow('image', stacked_image)
cv2.waitKey(0)
