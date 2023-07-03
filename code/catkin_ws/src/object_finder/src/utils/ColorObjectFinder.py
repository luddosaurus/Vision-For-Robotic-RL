import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class ColorObjectFinder:
    # HSV = Hue, Saturation, Value
    HUE_MAX = 179
    SAT_MAX = 255
    VAL_MAX = 255
    NOISE_MAX = 15
    FILL_MAX = 15

    HUE = 0
    SATURATION = 1
    VALUE = 2
    HUE_MARGIN = 3
    SATURATION_MARGIN = 4
    VALUE_MARGIN = 5

    HUE_MARGIN_MAX_CLICK = 40
    SAT_MARGIN_MAX_CLICK = 40
    VAL_MARGIN_MAX_CLICK = 40

    NOISE = 6
    FILL = 7

    # hue, sat, val , margin x 3, noise, fill
    saved_states = [
        [103, 224, 229, 40, 40, 40, 5, 10],
        [4, 165, 203, 13, 35, 54, 5, 4],  # orange
        [168, 177, 167, 26, 40, 76, 2, 3],  # red
        [104, 152, 150, 18, 81, 82, 3, 3],  # blue
        [25, 194, 225, 5, 89, 103, 5, 3],  # yellow
        [60, 107, 184, 21, 50, 74, 1, 3],  # green
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    current_state_index = 0

    def __init__(self) -> None:
        pass

    # Hue (degrees), Sat (percentage), Val (percentage)
    def set_color(self, hue, sat, val):
        self.saved_states[self.current_state_index][self.HUE] = self.HUE_MAX * (hue / 360)
        self.saved_states[self.current_state_index][self.SATURATION] = self.SAT_MAX * sat
        self.saved_states[self.current_state_index][self.VALUE] = self.VAL_MAX * val

    def set_margin(self, hue=0, sat=0, val=0, all_sliders=0):
        if all_sliders != 0:
            self.saved_states[self.current_state_index][self.HUE_MARGIN] = all_sliders
            self.saved_states[self.current_state_index][self.SATURATION_MARGIN] = all_sliders
            self.saved_states[self.current_state_index][self.VALUE_MARGIN] = all_sliders
        else:
            self.saved_states[self.current_state_index][self.HUE_MARGIN] = hue
            self.saved_states[self.current_state_index][self.SATURATION_MARGIN] = sat
            self.saved_states[self.current_state_index][self.VALUE_MARGIN] = val

    def remove_noise(self, mask, kernel_size):
        iterations = 3
        # kernel_size = self.saved_states[self.current_state_index][self.NOISE]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

        return cleaned_mask

    def fill_holes(self, mask, kernel_size):
        iterations = 3
        # kernel_size = self.saved_states[self.current_state_index][self.FILL]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        return filled_mask

    def get_hsv_mask(self, image, color_list=None):
        if color_list is None or len(color_list) == 0:
            current_states = [self.get_state()]

        else:
            current_states = color_list

        final_mask = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

        for current_state in current_states:
            # print(current_state)
            hue_range_l1 = current_state[self.HUE] - current_state[self.HUE_MARGIN]

            sat_range_l1 = current_state[self.SATURATION] - current_state[self.SATURATION_MARGIN]

            val_range_l1 = current_state[self.VALUE] - current_state[self.VALUE_MARGIN]

            lower_range_1 = np.array((
                hue_range_l1,
                sat_range_l1,
                val_range_l1
            ))

            hue_range_u1 = current_state[self.HUE] + current_state[self.HUE_MARGIN]

            sat_range_u1 = current_state[self.SATURATION] + current_state[self.SATURATION_MARGIN]

            val_range_u1 = current_state[self.VALUE] + current_state[self.VALUE_MARGIN]

            upper_range_1 = np.array((
                hue_range_u1,
                sat_range_u1,
                val_range_u1
            ))

            lower_range_2 = lower_range_1.copy()
            lower_range_2_compare = upper_range_1.copy()

            if hue_range_l1 < 0:
                lower_range_2[0] = self.HUE_MAX + hue_range_l1
                lower_range_2_compare[0] = self.HUE_MAX
            # if sat_range_l1 < 0:
            #     lower_range_2[1] = self.SAT_MAX + sat_range_l1
            #     lower_range_2_compare[1] = self.SAT_MAX
            # if val_range_l1 < 0:
            #     lower_range_2[2] = self.VAL_MAX + val_range_l1
            #     lower_range_2_compare[2] = self.VAL_MAX

            upper_range_2 = upper_range_1.copy()
            upper_range_2_compare = lower_range_1.copy()

            if hue_range_u1 > self.HUE_MAX:
                upper_range_2[0] = hue_range_u1 - self.HUE_MAX
                upper_range_2_compare[0] = 0
            # if sat_range_u1 > self.SAT_MAX:
            #     upper_range_2[1] = sat_range_u1 - self.SAT_MAX
            #     upper_range_2_compare[1] = 0
            # if val_range_u1 > self.VAL_MAX:
            #     upper_range_2[2] = val_range_u1 - self.VAL_MAX
            #     upper_range_2_compare[2] = 0

            # Convert the image to HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv_image, lower_range_1, upper_range_1)
            lower_wrap_mask = cv2.inRange(hsv_image, lower_range_2, lower_range_2_compare)
            upper_wrap_mask = cv2.inRange(hsv_image, upper_range_2_compare, upper_range_2)

            mask = cv2.bitwise_or(mask, lower_wrap_mask)
            mask = cv2.bitwise_or(mask, upper_wrap_mask)

            if current_state[self.FILL] > 0:
                mask = self.fill_holes(mask, current_state[self.FILL])

            if current_state[self.NOISE] > 0:
                mask = self.remove_noise(mask, current_state[self.NOISE])

            final_mask = cv2.bitwise_or(mask, final_mask)

        return final_mask

    @staticmethod
    def find_mask_center(mask):
        try:
            moments = cv2.moments(mask)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            return cx, cy

        except:
            return None, None

    @staticmethod
    def find_segment_coordinates(mask):
        components, outputs, stats, centroids = cv2.connectedComponentsWithStats(mask)

        segment_coordinates = []
        centroids = centroids[1:]  # remove
        for center in centroids:
            # coordinates = center[:, 0, :]  # Extract the x, y coordinates of the contour
            segment_coordinates.append([int(center_val) for center_val in center])

        return segment_coordinates

    @staticmethod
    def draw_dot(image, x, y, color=(255, 0, 255), radius=20, thickness=3):
        cv2.circle(image, (x, y), radius, color, thickness)

    @staticmethod
    def pixel_to_3d_coordinate(pixel_coord, depth_value, camera_matrix):
        # print(pixel_coord, depth_value, camera_matrix)
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        # print(depth_value)
        # Calculate the 3D coordinates
        x = (pixel_coord[0] - cx) * depth_value / fx
        y = (pixel_coord[1] - cy) * depth_value / fy
        z = depth_value

        return x, y, z

    def update_value(self, value, param):
        self.saved_states[self.current_state_index][param] = value

    def get_state(self):
        return self.saved_states[self.current_state_index]

    # def set_image_coordinate_color(self, image, x, y, roi_size, scale=1):
    #     x = int(x / scale)
    #     y = int(y / scale)
    #     print(image.shape)
    #     b, g, r = image[y, x]
    #     print(b, g, r)
    #     hsv = cv2.cvtColor(np.uint8([[(b, g, r)]]), cv2.COLOR_BGR2HSV)
    #     h, s, v = hsv[0][0]
    #     y_lower = max(y - roi_size, 0)
    #     y_upper = min(image.shape[0], y + roi_size)
    #     x_lower = max(x - roi_size, 0)
    #     x_upper = min(image.shape[1], x + roi_size)
    #     roi = image[y_lower: y_upper, x_lower: x_upper]
    #     # mean_color = np.mean(roi, axis=0)
    #     # print(mean_color)
    #     hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #     hsv_lower = np.min(hsv_roi, axis=(0, 1))
    #     hsv_upper = np.max(hsv_roi, axis=(0, 1))
    #     hue_diff_factor = 1
    #     sat_diff_factor = 1
    #     val_diff_factor = 1
    #     hue_diff = (hsv_upper[0] - hsv_lower[0]) * hue_diff_factor
    #     sat_diff = (hsv_upper[1] - hsv_lower[1]) * sat_diff_factor
    #     val_diff = (hsv_upper[2] - hsv_lower[2]) * val_diff_factor
    #
    #     self.update_value(h, self.HUE)
    #     self.update_value(s, self.SATURATION)
    #     self.update_value(v, self.VALUE)
    #     self.update_value(min(hue_diff, self.HUE_MARGIN_MAX_CLICK), self.HUE_MARGIN)
    #     self.update_value(min(sat_diff, self.SAT_MARGIN_MAX_CLICK), self.SATURATION_MARGIN)
    #     self.update_value(min(val_diff, self.VAL_MARGIN_MAX_CLICK), self.VALUE_MARGIN)
    #     print(f"h{h}, s{s}, v{v}")

    @staticmethod
    def get_image_coordinate_color(image, x, y, roi_size, scale=1):
        x = int(x / scale)
        y = int(y / scale)
        b, g, r = image[y, x]
        hsv = cv2.cvtColor(np.uint8([[(b, g, r)]]), cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[0][0]
        y_lower = max(y - roi_size, 0)
        y_upper = min(image.shape[0], y + roi_size)
        x_lower = max(x - roi_size, 0)
        x_upper = min(image.shape[1], x + roi_size)
        roi = image[y_lower: y_upper, x_lower: x_upper]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_clean = ColorObjectFinder.remove_outliers(hsv_roi)
        # hsv_lower = np.min(hsv_roi, axis=(0, 1))
        # hsv_upper = np.max(hsv_roi, axis=(0, 1))
        #
        # hue_diff = (hsv_upper[0] - hsv_lower[0]) / 2
        # sat_diff = (hsv_upper[1] - hsv_lower[1]) / 2
        # val_diff = (hsv_upper[2] - hsv_lower[2]) / 2
        return hsv_clean

    @staticmethod
    def remove_outliers(image):
        # Convert image to HSV color space
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split image into separate channels
        hue, saturation, value = cv2.split(image)
        data = {'hue': hue.flatten(), 'saturation': saturation.flatten(), 'value': value.flatten()}
        df = pd.DataFrame(data)
        plt.figure()
        df['hue'].hist(bins=179)
        plt.show()
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1

        # Do twice! Right shift lower values once
        # remove pixels with outliers
        true_list = ~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr)))
        subset = df[true_list]
        subset = subset.dropna()

        plt.figure()
        subset['hue'].hist(bins=179)
        plt.show()

        means = subset.mean()
        max_values = subset.max()
        min_values = subset.min()
        medians = subset.median()
        print(f'mean values:\n{means}\nmax values:\n{max_values}\nmin values:\n{min_values}\nmedian:\n{medians}')
        # # Remove outliers using z-score
        # z_scores = stats.zscore(hue.flatten())
        # hue = hue.flatten()[np.abs(z_scores) < 1]
        #
        # z_scores = stats.zscore(saturation.flatten())
        # saturation = saturation.flatten()[np.abs(z_scores) < 1]
        # # print(z_scores)
        # z_scores = stats.zscore(value.flatten())
        # value = value.flatten()[np.abs(z_scores) < 1]
        # print(hue, saturation, value)
        diff_hue = ColorObjectFinder.calculate_distance(min_values['hue'], max_values['hue'])
        diff_saturation = ColorObjectFinder.calculate_distance(min_values['saturation'], max_values['saturation'])
        diff_value = max_values['value'] - min_values['value']

        # if max['hue'] - min['hue'] > abs(min['hue'] - (179 - max['hue'])):
        #     total_sum = 0
        #     count = 0
        #     for hue_value in subset['hue']:
        #         if (179 - hue_value) < hue_value:
        #
        #             total_sum += 0 - (179 - hue_value)
        #         else:
        #
        #             total_sum += hue_value
        #         count += 1
        #     print('total_sum: ', total_sum/count)

        mean_hue = ColorObjectFinder.average(subset['hue'])
        print(mean_hue)

        # Calculate average hue, saturation, and value
        # avg_hue = int(np.mean(hue))
        # avg_saturation = int(np.mean(saturation))
        # avg_value = int(np.mean(value))

        return int(medians['hue']), int(means['saturation']), int(means['value']), int(diff_hue), int(
            diff_saturation), int(diff_value)

    @staticmethod
    def wrapped_hsv(image):

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)

        hist, bins = np.histogram(h.flatten(), bins=180, range=[0, 180])

        hue_values = list(hist)

        hue_values_mirror = [-val for val in hue_values]

        merged_hue_values = hue_values + hue_values_mirror

        merged_hue_values = [val for val in merged_hue_values if
                             np.percentile(merged_hue_values, 25) <= val <= np.percentile(merged_hue_values, 75)]

        average_value = np.mean(merged_hue_values)
        max_value = np.max(merged_hue_values)
        min_value = np.abs(np.min(merged_hue_values))

        return average_value, min_value, max_value

    @staticmethod
    def calculate_distance(value1, value2):
        span = 179
        distance = abs(value2 - value1)

        if distance > span / 2:
            distance = span - distance

        return distance

    @staticmethod
    def average(values):
        mi = min(values)
        ma = max(values)
        span = 179
        if ma - mi > span / 2:
            print('here')
            return (((mi + span) + ma) / 2.) % span
        else:
            print(mi, ma)
            return (mi + ma) / 2.

    def set_image_coordinate_color(self, image, x, y, roi_size, scale=1):

        hue, saturation, value, hue_diff, sat_diff, val_diff = self.get_image_coordinate_color(
            image, x, y, roi_size, scale)

        self.update_value(hue, self.HUE)
        self.update_value(saturation, self.SATURATION)
        self.update_value(value, self.VALUE)
        self.update_value(min(hue_diff, self.HUE_MARGIN_MAX_CLICK), self.HUE_MARGIN)
        self.update_value(min(sat_diff, self.SAT_MARGIN_MAX_CLICK), self.SATURATION_MARGIN)
        self.update_value(min(val_diff, self.VAL_MARGIN_MAX_CLICK), self.VALUE_MARGIN)

    def get_current_state(self):
        return self.saved_states[self.current_state_index]
