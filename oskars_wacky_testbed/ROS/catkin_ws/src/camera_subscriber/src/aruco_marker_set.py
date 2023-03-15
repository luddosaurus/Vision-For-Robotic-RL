import cv2
import numpy as np
import math


class MarkerSet(object):

    def __init__(self, corners, ids, marker_size, matrix_coefficients, distortion_coefficients, image):
        self.corners = corners
        self.ids = ids
        self.marker_size = marker_size
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients
        self.image = image
        self.distance_dict_triangle = {"0-1": 0,
                                       "0-2": 0,
                                       "1-2": 0}

    def draw_markers_with_axes(self):
        cv2.aruco.drawDetectedMarkers(self.image, self.corners)
        self.estimate_pose_of_markers()
        return self.image

    def get_camera_distance_to_markers_via_transform(self):
        distances = dict()
        for i in range(0, len(self.ids)):
            print(f'one print: {len(self.ids)} - {i}')
            rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(self.corners[i], self.marker_size,
                                                                            self.matrix_coefficients,
                                                                            self.distortion_coefficients)
            cv2.drawFrameAxes(self.image, self.matrix_coefficients, self.distortion_coefficients, rvec, tvec,
                              self.marker_size / 2)

            distances[self.ids[i][0]] = tvec[0][0][2]
        return distances

    def get_center_of_every_marker(self):
        centers = dict()
        for i in range(0, len(self.ids)):
            center = self.find_center(mark_corners=self.corners[i].reshape((4, 2)))
            centers[self.ids[i][0]] = center
        return centers

    def estimate_pose_of_markers(self):
        for i in range(0, len(self.ids)):
            rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(self.corners[i], self.marker_size,
                                                                            self.matrix_coefficients,
                                                                            self.distortion_coefficients)
            cv2.drawFrameAxes(self.image, self.matrix_coefficients, self.distortion_coefficients, rvec, tvec,
                              self.marker_size / 2)

    def calc_distance_from_each(self):
        cv2.aruco.drawDetectedMarkers(self.image, self.corners)
        for i in range(0, len(self.ids)):

            rvec_i, tvec_i, markerPoints_i = cv2.aruco.estimatePoseSingleMarkers(self.corners[i], self.marker_size,
                                                                                 self.matrix_coefficients,
                                                                                 self.distortion_coefficients)
            c_i = self.find_center(self.corners[i].reshape((4, 2)))

            for j in range(i + 1, len(self.ids)):
                rvec_j, tvec_j, markerPoints_j = cv2.aruco.estimatePoseSingleMarkers(self.corners[j], self.marker_size,
                                                                                     self.matrix_coefficients,
                                                                                     self.distortion_coefficients)

                distance = self.calculate_distance_between_two_markers(tvec_i, tvec_j)
                # print(f'{i} - {j}: {diff}')

                c_j = self.find_center(self.corners[j].reshape((4, 2)))
                cv2.line(self.image, c_i, c_j, (147, 20, 255), 2)

                cv2.putText(self.image, f"{distance:.2f}",
                            (int(abs(c_i[0] + c_j[0]) / 2), int(abs(c_i[1] + c_j[1]) / 2)), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 2, cv2.LINE_AA, )
                cv2.drawFrameAxes(self.image, self.matrix_coefficients, self.distortion_coefficients, rvec_j, tvec_j,
                                  3.4 / 2)

            cv2.drawFrameAxes(self.image, self.matrix_coefficients, self.distortion_coefficients, rvec_i, tvec_i,
                              3.4 / 2)

        return self.image

    def calc_distance_from_each_area_version(self):
        cv2.aruco.drawDetectedMarkers(self.image, self.corners)

        for i in range(0, len(self.ids)):

            rvec_i, tvec_i, markerPoints_i = cv2.aruco.estimatePoseSingleMarkers(self.corners[i], self.marker_size,
                                                                                 self.matrix_coefficients,
                                                                                 self.distortion_coefficients)
            c_i = self.find_center(self.corners[i].reshape((4, 2)))
            # test = np.zeros((4,4))
            r_mat, _ = cv2.Rodrigues(rvec_i)
            cam_r = np.transpose(r_mat)
            cam_r_vec, _ = cv2.Rodrigues(r_mat)
            cam_t_vec = np.matmul(-cam_r, np.reshape(tvec_i, (3, 1)))
            # print(f'{cam_t_vec}\n-----\n{tvec_i}')

            for j in range(i + 1, len(self.ids)):
                rvec_j, tvec_j, markerPoints_j = cv2.aruco.estimatePoseSingleMarkers(self.corners[j], self.marker_size,
                                                                                     self.matrix_coefficients,
                                                                                     self.distortion_coefficients)

                distance = self.calculate_distance_between_two_markers(tvec_i, tvec_j)

                self.distance_dict_triangle[f'{i}-{j}'] = distance
                # print(f'{i} - {j}: {diff}')

                c_j = self.find_center(self.corners[j].reshape((4, 2)))
                cv2.line(self.image, c_i, c_j, (147, 20, 255), 2)

                cv2.putText(self.image, f"{distance:.2f}",
                            (int(abs(c_i[0] + c_j[0]) / 2), int(abs(c_i[1] + c_j[1]) / 2)), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 2, cv2.LINE_AA, )
                cv2.drawFrameAxes(self.image, self.matrix_coefficients, self.distortion_coefficients, rvec_j, tvec_j,
                                  3.4 / 2)

            cv2.drawFrameAxes(self.image, self.matrix_coefficients, self.distortion_coefficients, rvec_i, tvec_i,
                              3.4 / 2)

        # print(self.calculate_area_between_3_markers(distance_dict["0-1"], distance_dict["0-2"], distance_dict["1-2"]))
        if len(self.corners) == 3:
            area = self.calculate_triangle_area(self.distance_dict_triangle["0-1"], self.distance_dict_triangle["0-2"],
                                                self.distance_dict_triangle["1-2"])
            cv2.putText(self.image, f"Area: {area:.2f}", (int(abs(c_i[0] + c_j[0]) / 2), int(abs(c_i[1] + c_j[1]) / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA, )
        return self.image

    def calculate_triangle_area(self, a, b, c):
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    def calculate_distance_between_two_markers(self, mark_1_tvec, mark_2_tvec):
        return np.linalg.norm(mark_1_tvec - mark_2_tvec)

    # def calculate_area_between_3_markers(self, mark_1_tvec, mark_2_tvec, mark_3_tvec):
    #      distance_1_2 = self.calculate_distance_between_two_markers(mark_1_tvec, mark_2_tvec)
    #      distance_1_3 = self.calculate_distance_between_two_markers(mark_1_tvec, mark_3_tvec)
    #      distance_2_3 = self.calculate_distance_between_two_markers(mark_2_tvec, mark_3_tvec)

    #      angle_1_2 = self.get_angle_with_cosine_law(distance_1_2, distance_1_3, distance_2_3)

    #      area = self.area_law(distance_1_2, distance_1_3, angle_1_2)

    #      return area

    # # alpha = cos^-1((b^2 + c^2 - a^2)/(2bc))
    # def get_angle_with_cosine_law(self, side_1, side_2, side_3):
    #     if side_1 >= 1 and side_2 >= 1:
    #         angle = math.acos((side_1**2 + side_2**2 - side_3**2)/(2*side_1*side_2))
    #     else:
    #         angle = 0
    #     return angle

    # def area_law(self, side_1, side_2, angle):
    #      return (side_1*side_2*math.sin(angle))/2

    def find_center(self, mark_corners):
        (topLeft, topRight, bottomRight, bottomLeft) = mark_corners

        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)

        return cX, cY
