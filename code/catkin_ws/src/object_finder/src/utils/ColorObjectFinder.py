import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import circmean, circstd, circvar

from utils.Const import Const


class ColorObjectFinder:
    saved_state = [103, 224, 229, 40, 40, 40, 1, 9]
    def __init__(self) -> None:
        # hue, sat, val , margin x 3, noise, fill
        pass

    # Hue (degrees), Sat (percentage), Val (percentage)
    def set_color(self, hue, sat, val):
        self.saved_state[Const.HUE] = Const.HUE_MAX * (hue / 360)
        self.saved_state[Const.SATURATION] = Const.SAT_MAX * sat
        self.saved_state[Const.VALUE] = Const.VAL_MAX * val

    def set_margin(self, hue=0, sat=0, val=0, all_sliders=0):
        if all_sliders != 0:
            self.saved_state[Const.HUE_MARGIN] = all_sliders
            self.saved_state[Const.SATURATION_MARGIN] = all_sliders
            self.saved_state[Const.VALUE_MARGIN] = all_sliders
        else:
            self.saved_state[Const.HUE_MARGIN] = hue
            self.saved_state[Const.SATURATION_MARGIN] = sat
            self.saved_state[Const.VALUE_MARGIN] = val

    def remove_noise(self, mask, kernel_size):
        iterations = 3
        # kernel_size = self.saved_states[self.NOISE]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

        return cleaned_mask

    def fill_holes(self, mask, kernel_size):
        iterations = 3
        # kernel_size = self.saved_states[self.FILL]

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
            hue_range_l1 = current_state[Const.HUE] - current_state[Const.HUE_MARGIN]

            sat_range_l1 = current_state[Const.SATURATION] - current_state[Const.SATURATION_MARGIN]

            val_range_l1 = current_state[Const.VALUE] - current_state[Const.VALUE_MARGIN]

            lower_range_1 = np.array((
                hue_range_l1,
                sat_range_l1,
                val_range_l1
            ))

            hue_range_u1 = current_state[Const.HUE] + current_state[Const.HUE_MARGIN]

            sat_range_u1 = current_state[Const.SATURATION] + current_state[Const.SATURATION_MARGIN]

            val_range_u1 = current_state[Const.VALUE] + current_state[Const.VALUE_MARGIN]

            upper_range_1 = np.array((
                hue_range_u1,
                sat_range_u1,
                val_range_u1
            ))

            lower_range_2 = lower_range_1.copy()
            lower_range_2_compare = upper_range_1.copy()

            if hue_range_l1 < 0:
                lower_range_2[0] = Const.HUE_MAX + hue_range_l1
                lower_range_2_compare[0] = Const.HUE_MAX
            # if sat_range_l1 < 0:
            #     lower_range_2[1] = Const.SAT_MAX + sat_range_l1
            #     lower_range_2_compare[1] = Const.SAT_MAX
            # if val_range_l1 < 0:
            #     lower_range_2[2] = Const.VAL_MAX + val_range_l1
            #     lower_range_2_compare[2] = Const.VAL_MAX

            upper_range_2 = upper_range_1.copy()
            upper_range_2_compare = lower_range_1.copy()

            if hue_range_u1 > Const.HUE_MAX:
                upper_range_2[0] = hue_range_u1 - Const.HUE_MAX
                upper_range_2_compare[0] = 0
            # if sat_range_u1 > Const.SAT_MAX:
            #     upper_range_2[1] = sat_range_u1 - Const.SAT_MAX
            #     upper_range_2_compare[1] = 0
            # if val_range_u1 > Const.VAL_MAX:
            #     upper_range_2[2] = val_range_u1 - Const.VAL_MAX
            #     upper_range_2_compare[2] = 0

            # Convert the image to HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv_image, lower_range_1, upper_range_1)
            lower_wrap_mask = cv2.inRange(hsv_image, lower_range_2, lower_range_2_compare)
            upper_wrap_mask = cv2.inRange(hsv_image, upper_range_2_compare, upper_range_2)

            mask = cv2.bitwise_or(mask, lower_wrap_mask)
            mask = cv2.bitwise_or(mask, upper_wrap_mask)

            if current_state[Const.FILL] > 0:
                mask = self.fill_holes(mask, current_state[Const.FILL])

            if current_state[Const.NOISE] > 0:
                mask = self.remove_noise(mask, current_state[Const.NOISE])

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
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        segment_coordinates = []
        centroids = centroids[1:]  # remove first one that is the whole image
        for center in centroids:
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
        self.saved_state[param] = value

    def get_state(self):
        return self.saved_state

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
        # plt.figure()
        # df['hue'].hist(bins=179)
        # plt.show()
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1

        # Do twice! Right shift lower values once
        # remove pixels with outliers
        true_list = ~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr)))
        subset = df[true_list]
        subset = subset.dropna()
        #
        # plt.figure()
        # subset['hue'].hist(bins=179)
        # plt.show()

        means = subset.mean()
        max_values = subset.max()
        min_values = subset.min()
        medians = subset.median()
        print(f'mean values:\n{means}'
              f'\nmax values:\n{max_values}'
              f'\nmin values:\n{min_values}'
              f'\nmedian:\n{medians}'
              )
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
        # diff_hue = ColorObjectFinder.calculate_distance(min_values['hue'], max_values['hue'])
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

        # mean_hue = ColorObjectFinder.average(subset['hue'])
        # print(mean_hue)

        # Calculate average hue, saturation, and value
        # avg_hue = int(np.mean(hue))
        # avg_saturation = int(np.mean(saturation))
        # avg_value = int(np.mean(value))

        hue_avg, hue_diff = ColorObjectFinder.wrapped_hue(image)

        print(hue_avg, hue_diff)

        return round(hue_avg), round(means['saturation']), round(means['value']), round(hue_diff), round(
            diff_saturation), round(diff_value)

    @staticmethod
    def wrapped_hue(image):
        # roi = 100
        # image = cv2.cvtColor(image[:roi, :roi], cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(image)
        hue_list = list(hue.flatten())

        # Remove circular outliers
        hue_mean = circmean(hue_list, high=179, low=0)
        hue_std = circstd(hue_list, high=179, low=0)
        hue_var = circvar(hue_list, high=179, low=0)
        # print(hue_mean)
        # print(hue_std)
        # print(hue_var)

        # Remove circular outliers based on quantiles
        q1 = np.percentile(hue_list, 25)
        q4 = np.percentile(hue_list, 75)
        hue_filtered = [val for val in hue_list if q1 <= val <= q4]
        # todo try variance on hue_filtered
        hue_diff = hue_var * 179

        return hue_mean, hue_diff

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

        self.update_value(hue, Const.HUE)
        self.update_value(saturation, Const.SATURATION)
        self.update_value(value, Const.VALUE)
        self.update_value(min(hue_diff, Const.HUE_MARGIN_MAX_CLICK), Const.HUE_MARGIN)
        self.update_value(min(sat_diff, Const.SAT_MARGIN_MAX_CLICK), Const.SATURATION_MARGIN)
        self.update_value(min(val_diff, Const.VAL_MARGIN_MAX_CLICK), Const.VALUE_MARGIN)

    def get_current_state(self):
        return self.saved_state.copy()
