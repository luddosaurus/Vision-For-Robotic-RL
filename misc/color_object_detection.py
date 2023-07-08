import cv2
import numpy as np
import DaVinci
import math


class ColorObjectDetector:
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
    NOISE = 6
    FILL = 7

    # hue, sat, val , margin, noise, fill
    saved_states = [
        [103, 224, 229, 40, 40, 40, 5, 10],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
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

    def set_margin(self, hue=0, sat=0, val=0, all=0):
        if all != 0:
            self.saved_states[self.current_state_index][self.HUE_MARGIN] = all
            self.saved_states[self.current_state_index][self.SATURATION_MARGIN] = all
            self.saved_states[self.current_state_index][self.VALUE_MARGIN] = all
        else:
            self.saved_states[self.current_state_index][self.HUE_MARGIN] = hue
            self.saved_states[self.current_state_index][self.SATURATION_MARGIN] = sat
            self.saved_states[self.current_state_index][self.VALUE_MARGIN] = val

    def remove_noise(self, mask):
        iterations = 3
        kernel_size = self.saved_states[self.current_state_index][self.NOISE]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

        return cleaned_mask

    def fill_holes(self, mask):
        iterations = 3
        kernel_size = self.saved_states[self.current_state_index][self.FILL]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        return filled_mask

    def find_box_orientation(self, mask):
        # # Find contours in the mask
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Filter out small contours
        # min_contour_area = 100
        # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # # Fit rotated rectangles to the contours
        # orientations = []
        # for contour in contours:
        #     rect = cv2.minAreaRect(contour)
        #     angle = rect[2]
        #     orientations.append(angle)

        return "hej"

    def get_hsv_mask(self, image):
        current_state = self.get_state()

        lower_range = np.array((
            current_state[cd.HUE] - current_state[cd.HUE_MARGIN],
            current_state[cd.SATURATION] - current_state[cd.SATURATION_MARGIN],
            current_state[cd.VALUE] - current_state[cd.VALUE_MARGIN]
        ))

        upper_range = np.array((
            current_state[cd.HUE] + current_state[cd.HUE_MARGIN],
            current_state[cd.SATURATION] + current_state[cd.SATURATION_MARGIN],
            current_state[cd.VALUE] + current_state[cd.VALUE_MARGIN]
        ))

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_image, lower_range, upper_range)

        if current_state[cd.FILL] != 0:
            mask = cd.fill_holes(mask)

        if current_state[cd.NOISE] != 0:
            mask = cd.remove_noise(mask)

        return mask

    def find_mask_center(self, mask):
        try:
            moments = cv2.moments(mask)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            return cx, cy

        except:
            return None, None

    def draw_dot(self, image, x, y, color=(255, 0, 255), radius=20, thickness=3):
        cv2.circle(image, (x, y), radius, color, thickness)

    def pixel_to_3d_coordinate(self, pixel_coord, depth_value, camera_matrix):
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Calculate the 3D coordinates
        x = (pixel_coord[0] - cx) * depth_value / fx
        y = (pixel_coord[1] - cy) * depth_value / fy
        z = depth_value

        return x, y, z

    def update_value(self, value, param):
        self.saved_states[self.current_state_index][param] = value

    def get_state(self):
        return self.saved_states[self.current_state_index]

    def set_coordinate_color(self, image, x, y):
        r = 10

        b, g, r = image[y, x]
        hsv = cv2.cvtColor(np.uint8([[(b, g, r)]]), cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0][0]

        roi = image[y - r:y + r, x - r:x + r]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_lower = np.min(hsv_roi, axis=(0, 1))
        hsv_upper = np.max(hsv_roi, axis=(0, 1))

        hue_diff = (hsv_upper[0] - hsv_lower[0]) * 3
        sat_diff = (hsv_upper[1] - hsv_lower[1]) * 3
        val_diff = (hsv_upper[2] - hsv_lower[2]) * 3

        self.update_value(h, cd.HUE)
        self.update_value(s, cd.SATURATION)
        self.update_value(v, cd.VALUE)
        self.update_value(hue_diff, cd.HUE_MARGIN)
        self.update_value(sat_diff, cd.SATURATION_MARGIN)
        self.update_value(val_diff, cd.VALUE_MARGIN)
        print(f"h{h}, s{s}, v{v}")


def draw_text_box(
        image,
        text,
        position="bottom_left",
        background=(255, 0, 255),
        foreground=(255, 255, 255),
        font_scale=1.0,
        thickness=2,
        box=True,
        margin=40):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the image dimensions
    height, width, _ = image.shape

    # Get the text size
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Set the position coordinates based on the input position
    if position == 'top_left':
        x = margin
        y = margin
    elif position == 'top_right':
        x = width - margin - text_size[0]
        y = margin
    elif position == 'bottom_left':
        x = margin
        y = height - margin
    elif position == 'bottom_right':
        x = width - margin - text_size[0]
        y = height - margin
    else:
        x = width / 2
        y = height / 2

    if box:
        # Calculate the box coordinates
        box_x1 = x
        box_y1 = y - text_size[1] - 20
        box_x2 = x + text_size[0] + 20
        box_y2 = y

        # Draw the rectangle as the box background
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), background, -1)

        # Put the text inside the box
        cv2.putText(image, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, foreground, thickness,
                    cv2.LINE_AA)

    else:
        # Border

        # Set the outline parameters
        outline_thickness = 3
        outline_color = background
        # Draw the outline text
        for dx in range(-outline_thickness, outline_thickness + 1):
            for dy in range(-outline_thickness, outline_thickness + 1):
                cv2.putText(image, text, (x + dx, y + dy), font, font_scale, outline_color, thickness)
        # Draw the main text
        cv2.putText(image, text, (x, y), font, font_scale, foreground, thickness)

    return image


def find_contour_center(contour):
    # Calculate the moments of the contour
    M = cv2.moments(contour)

    # Calculate the center of mass
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    return center_x, center_y


def draw_pose_vectors(image, rotation_matrix, tvec, camera_matrix, dist_coeffs):
    # Define the 3D object points in the local coordinate system
    object_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    # Project the 3D object points onto the image plane
    image_points, _ = cv2.projectPoints(object_points, rotation_matrix, tvec, camera_matrix, dist_coeffs)

    # Draw the 3D pose on the image
    origin = tuple(image_points[0].ravel().astype(int))
    x_axis = tuple(image_points[1].ravel().astype(int))
    y_axis = tuple(image_points[2].ravel().astype(int))
    z_axis = tuple(image_points[3].ravel().astype(int))

    # Draw the coordinate axes
    cv2.line(image, tuple(map(int, origin)), tuple(map(int, x_axis)), (0, 0, 255), 2)  # Red for X-axis
    cv2.line(image, tuple(map(int, origin)), tuple(map(int, y_axis)), (0, 255, 0), 2)  # Green for Y-axis
    cv2.line(image, tuple(map(int, origin)), tuple(map(int, z_axis)), (255, 0, 0), 2)  # Blue for Z-axis

    return image


def draw_rotated_boxes(image, mask):
    # Find contours of the segments in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rotated bounding boxes around the segments
    for contour in contours:
        # Find the minimum area rectangle enclosing the contour
        rect = cv2.minAreaRect(contour)

        box = np.intp(cv2.boxPoints(rect))

        rmat, tvec = estimate_3d_pose(box, camera_matrix, distortion_coefficients, None)

        draw_pose_vectors(image, rmat, tvec, camera_matrix, distortion_coefficients)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    return image


def estimate_3d_pose(box, camera_matrix, dist_coeffs, depth_image):
    # Define the 2D image points of the box
    image_points = np.array(box, dtype=np.float32)

    object_points = np.array([cd.pixel_to_3d_coordinate((x, y), 0.53, camera_matrix) for x, y in box])
    print(object_points)

    # Estimate the rotation and translation vectors
    _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    print(tvec)
    return rotation_matrix, tvec


def update_trackbars():
    current_state = cd.get_state()
    cv2.setTrackbarPos("Hue", window, current_state[cd.HUE])
    cv2.setTrackbarPos("Saturation", window, current_state[cd.SATURATION])
    cv2.setTrackbarPos("Value", window, current_state[cd.VALUE])
    cv2.setTrackbarPos("Hue Margin", window, current_state[cd.HUE_MARGIN])
    cv2.setTrackbarPos("Sat Margin", window, current_state[cd.SATURATION_MARGIN])
    cv2.setTrackbarPos("Val Margin", window, current_state[cd.VALUE_MARGIN])
    cv2.setTrackbarPos("Noise", window, current_state[cd.NOISE])
    cv2.setTrackbarPos("Fill", window, current_state[cd.FILL])


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("click!")
        cd.set_coordinate_color(image, x, y)
        update_trackbars()


# ------------------------------ Main
window = 'ColorDetection'
cv2.namedWindow(window)

cap = cv2.VideoCapture(0)
cd = ColorObjectDetector()

start_state = cd.get_state()
# cv2.createTrackbar("Hue", window, start_state[cd.HUE], cd.HUE_MAX, lambda value: cd.update_value(value, cd.HUE))
# cv2.createTrackbar("Saturation", window, start_state[cd.SATURATION], cd.SAT_MAX, lambda value: cd.update_value(value, cd.SATURATION))
# cv2.createTrackbar("Value", window, start_state[cd.VALUE], cd.VAL_MAX, lambda value: cd.update_value(value, cd.VALUE))
#
# cv2.createTrackbar("Hue Margin", window, start_state[cd.HUE_MARGIN], cd.HUE_MAX, lambda value: cd.update_value(value, cd.HUE_MARGIN))
# cv2.createTrackbar("Sat Margin", window, start_state[cd.SATURATION_MARGIN], cd.SAT_MAX, lambda value: cd.update_value(value, cd.SATURATION_MARGIN))
# cv2.createTrackbar("Val Margin", window, start_state[cd.VALUE_MARGIN], cd.VAL_MAX, lambda value: cd.update_value(value, cd.VALUE_MARGIN))
#
# cv2.createTrackbar("Noise", window, start_state[cd.NOISE], cd.NOISE_MAX, lambda value: cd.update_value(value, cd.NOISE))
# cv2.createTrackbar("Fill", window, start_state[cd.FILL], cd.FILL_MAX, lambda value: cd.update_value(value, cd.FILL))

image = None
# cv2.setMouseCallback(window, click)

pose_esitmate = False
if pose_esitmate:
    with np.load("cv/intrinsic_matrix.npz") as X:
        camera_matrix, distortion_coefficients, _, _ = \
            [X[i] for i in ('matrix', 'distortion', 'rotation_vectors', 'translation_vectors')]

img_dir = '/home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/misc/assets'
single_image = True
if single_image:
    image = cv2.imread(img_dir + '/legos_4.jpg')

    mask_image = cd.get_hsv_mask(image=image)
    res = cv2.bitwise_and(image, image, mask=mask_image)
    mask = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    box_image = draw_rotated_boxes(image, mask_image)

    # Find center
    x, y = cd.find_mask_center(mask_image)
    pose_info = ""
    if x is not None:
        cd.draw_dot(res, x, y)

        # Find 3D point
        if pose_esitmate:
            depth = 0.4
            position = cd.pixel_to_3d_coordinate((x, y), depth, camera_matrix)
            pose_info = f"x{position[0]:.2f} : y{position[1]:.2f}, z{position[2]:.2f}"
    rotated_image = cv2.rotate(image, rotateCode=2)
    resized_image = cv2.resize(rotated_image, None, fx=0.2, fy=0.2)
    rotated_res = cv2.rotate(res, rotateCode=2)
    resized_res = cv2.resize(rotated_res, None, fx=0.2, fy=0.2)
    rotated_box_image = cv2.rotate(box_image, rotateCode=2)
    resized_box_image = cv2.resize(rotated_box_image, None, fx=0.2, fy=0.2)
    # Show Image
    stacked = np.hstack((resized_image, resized_res, resized_box_image))

    info = "[0-9] states, [m]ove to, [q]uit"
    draw_text_box(
        image=stacked,
        text=info
    )

    slot_info = f"Color State [{cd.current_state_index}]"
    draw_text_box(
        image=stacked,
        text=slot_info,
        position="top_left"
    )

    if pose_esitmate and pose_info != "":
        draw_text_box(
            image=stacked,
            text=pose_info,
            position="top_right"
        )

    cv2.imshow(window, stacked)
    cv2.waitKey(0)
else:
    while True:
        # Get Image
        ret, image = cap.read()
        if not ret:
            break

        # Mask
        mask_image = cd.get_hsv_mask(image=image)
        res = cv2.bitwise_and(image, image, mask=mask_image)
        mask = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

        box_image = draw_rotated_boxes(image, mask_image)

        # Find center
        x, y = cd.find_mask_center(mask_image)
        pose_info = ""
        if x is not None:
            cd.draw_dot(res, x, y)

            # Find 3D point
            if pose_esitmate:
                depth = 0.4
                position = cd.pixel_to_3d_coordinate((x, y), depth, camera_matrix)
                pose_info = f"x{position[0]:.2f} : y{position[1]:.2f}, z{position[2]:.2f}"

        # Show Image
        stacked = np.hstack((image, res, box_image))

        info = "[0-9] states, [m]ove to, [q]uit"
        draw_text_box(
            image=stacked,
            text=info
        )

        slot_info = f"Color State [{cd.current_state_index}]"
        draw_text_box(
            image=stacked,
            text=slot_info,
            position="top_left"
        )

        if pose_esitmate and pose_info != "":
            draw_text_box(
                image=stacked,
                text=pose_info,
                position="top_right"
            )
        scale = 0.5
        cv2.imshow(window, cv2.resize(stacked, None, fx=scale, fy=scale))

        # Input
        key = cv2.waitKey(1) & 0xFF
        key_str = chr(key)

        if key_str.isdigit() and 0 <= int(key_str) <= 9:
            key_number = int(key_str)
            cd.current_state_index = key_number
            update_trackbars()
            print(f"Switching to {key_number}")

        elif key == ord('m'):
            print("Going to...")

        elif key == ord('q'):
            print("Bye :)")
            break

cv2.destroyAllWindows()
