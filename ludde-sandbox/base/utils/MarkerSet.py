import cv2
import numpy as np

class MarkerSet(object):

    def __init__(self, corners, ids, marker_size, matrix_coefficients, distortion_coefficients, image):
        self.corners = corners
        self.ids = ids
        self.marker_size = marker_size
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients
        self.image = image

    def calc_distance_from_each(self):
        cv2.aruco.drawDetectedMarkers(self.image, self.corners)
        for i in range(0, len(self.ids)):

            rvec_i, tvec_i, markerPoints_i = cv2.aruco.estimatePoseSingleMarkers(self.corners[i], self.marker_size,
                                                                                 self.matrix_coefficients,
                                                                                 self.distortion_coefficients)
            c_i = self.find_center(i)

            for j in range(i + 1, len(self.ids)):
                rvec_j, tvec_j, markerPoints_j = cv2.aruco.estimatePoseSingleMarkers(self.corners[j], self.marker_size,
                                                                                     self.matrix_coefficients,
                                                                                     self.distortion_coefficients)

                diff = np.linalg.norm(tvec_j - tvec_i)
                # print(f'{i} - {j}: {diff}')

                c_j = self.find_center(j)
                cv2.line(self.image, c_i, c_j, (147, 20, 255), 2)

                cv2.putText(bgr_color_image, f"{diff:.2f}",
                            (int(abs(c_i[0] + c_j[0]) / 2), int(abs(c_i[1] + c_j[1]) / 2)), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 2, cv2.LINE_AA, )
                cv2.drawFrameAxes(self.image, self.matrix_coefficients, self.distortion_coefficients, rvec_j, tvec_j,
                                  3.4 / 2)

            cv2.drawFrameAxes(self.image, self.matrix_coefficients, self.distortion_coefficients, rvec_i, tvec_i,
                              3.4 / 2)

        return self.image

    def find_center(self, index):
        corners = self.corners[index].reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)

        return cX, cY

    def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

        marker_size_cm = 3.4
        if len(corners) > 0:
            markerSet = MarkerSet(corners, ids, marker_size_cm, matrix_coefficients, distortion_coefficients, frame)
            frame = markerSet.calc_distance_from_each()

            # for i in range(0, len(ids)):
            #     marker_size_cm = 3.4
            #     rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size_cm, matrix_coefficients,
            #                                                                distortion_coefficients)

            #     cv2.aruco.drawDetectedMarkers(frame, corners)

            #     cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 3.4/2)
            # cv2.putText(bgr_color_image, f"{tvec[0][0][2]}", (int(corners[0][0][1][0]),int(corners[0][0][1][1]) ),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),2,cv2.LINE_AA,)

        return frame