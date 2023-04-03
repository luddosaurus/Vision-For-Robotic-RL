import cv2
from cv2 import aruco

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def paint_dots(image, coordinates):

    color = (0, 255, 0)

    for coordinate in coordinates:
        cv2.circle(image, (coordinate[0], coordinate[1]), 3, color, -1)


def draw_vectors(img, marker_corners, marker_ids, matrix, distortion):

    marker_size = 0.05
    axis_length = 0.02
    if len(marker_corners) > 0:
        for i in range(0, len(marker_ids)):
            rotation_vec, translation_vec, marker_points = aruco.estimatePoseSingleMarkers(
                corners=marker_corners[i],
                markerLength=0.05,
                cameraMatrix=matrix,
                distCoeffs=distortion
            )
            if rotation_vec is not None:
                #print("Rotation: ", rotation_vec)
                # z - blue , y - green, x - red
                cv2.drawFrameAxes(
                    image=img,
                    cameraMatrix=matrix,
                    distCoeffs=distortion,
                    rvec=rotation_vec,
                    tvec=translation_vec,
                    length=axis_length)

    return img


def plot_3d_points_and_vector(points, vector):
    # create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the 3D points
    points = np.array(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, c='r', marker='o')

    # plot the 3D vector
    p, v = vector
    v = np.array(v)
    v /= np.linalg.norm(v)  # normalize the vector
    ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], length=1.0, color='g')

    # add labels to the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # show the plot
    plt.show()