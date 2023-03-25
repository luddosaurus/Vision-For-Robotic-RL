import cv2

# Handle all the drawings on the images


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