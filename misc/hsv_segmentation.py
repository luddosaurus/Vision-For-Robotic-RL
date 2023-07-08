import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

img_dir = '/home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/misc/assets'
image = cv2.imread(img_dir + '/legos_red.jpg')

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

# plt.imshow(hsv_image)
# plt.show()
#
pixel_colors = rgb_image.reshape((np.shape(rgb_image)[0] * np.shape(rgb_image)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
#
# r, g, b = cv2.split(rgb_image)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")
#
# axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Red")
# axis.set_ylabel("Green")
# axis.set_zlabel("Blue")
# plt.show()
#
h, s, v = cv2.split(hsv_image)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
# plt.imshow(hsv_image)
# plt.show()
