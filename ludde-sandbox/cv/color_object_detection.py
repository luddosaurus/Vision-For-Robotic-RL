import cv2
import numpy as np
import DaVinci

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
        self.saved_states[self.current_state_index][self.HUE] = self.HUE_MAX*(hue/360)
        self.saved_states[self.current_state_index][self.SATURATION] = self.SAT_MAX*sat 
        self.saved_states[self.current_state_index][self.VALUE] = self.VAL_MAX*val

        
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

        # Set the position coordinates based on the input position
        if position == 'top_left':
            x = margin
            y = margin
        elif position == 'top_right':
            x = width - margin
            y = margin
        elif position == 'bottom_left':
            x = margin
            y = height - margin
        elif position == 'bottom_right':
            x = width - margin
            y = height - margin
        else:
            x = width / 2
            y = height / 2

        # Get the text size
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

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

def update_trackbars(window_name):
    current_state = cd.get_state()
    cv2.setTrackbarPos("Hue", window_name, current_state[cd.HUE])
    cv2.setTrackbarPos("Saturation", window_name, current_state[cd.SATURATION])
    cv2.setTrackbarPos("Value", window_name, current_state[cd.VALUE])
    cv2.setTrackbarPos("Hue Margin", window_name, current_state[cd.HUE_MARGIN])
    cv2.setTrackbarPos("Sat Margin", window_name, current_state[cd.SATURATION_MARGIN])
    cv2.setTrackbarPos("Val Margin", window_name, current_state[cd.VALUE_MARGIN])
    cv2.setTrackbarPos("Noise", window_name, current_state[cd.NOISE])
    cv2.setTrackbarPos("Fill", window_name, current_state[cd.FILL])

# ------------------------------ Main
window = 'ColorDetection'
cv2.namedWindow(window)

cap = cv2.VideoCapture(0)
cd = ColorObjectDetector()

start_state = cd.get_state()
cv2.createTrackbar("Hue", window, start_state[cd.HUE], cd.HUE_MAX, lambda value: cd.update_value(value, cd.HUE))
cv2.createTrackbar("Saturation", window, start_state[cd.SATURATION], cd.SAT_MAX, lambda value: cd.update_value(value, cd.SATURATION))
cv2.createTrackbar("Value", window, start_state[cd.VALUE], cd.VAL_MAX, lambda value: cd.update_value(value, cd.VALUE))

cv2.createTrackbar("Hue Margin", window, start_state[cd.HUE_MARGIN], cd.HUE_MAX, lambda value: cd.update_value(value, cd.HUE_MARGIN))
cv2.createTrackbar("Sat Margin", window, start_state[cd.SATURATION_MARGIN], cd.SAT_MAX, lambda value: cd.update_value(value, cd.SATURATION_MARGIN))
cv2.createTrackbar("Val Margin", window, start_state[cd.VALUE_MARGIN], cd.VAL_MAX, lambda value: cd.update_value(value, cd.VALUE_MARGIN))

cv2.createTrackbar("Noise", window, start_state[cd.NOISE], cd.NOISE_MAX, lambda value: cd.update_value(value, cd.NOISE))
cv2.createTrackbar("Fill", window, start_state[cd.FILL], cd.FILL_MAX, lambda value: cd.update_value(value, cd.FILL))

# with np.load("cv/intrinsic_matrix.npz") as X:
#             camera_matrix, distortion_coefficients, _, _ = \
#                 [X[i] for i in ('matrix', 'distortion', 'rotation_vectors', 'translation_vectors')]

while True:
    # Get Image
    ret, image = cap.read()
    if not ret:
        break

    # Mask
    mask_image = cd.get_hsv_mask(image=image)
    res = cv2.bitwise_and(image, image, mask=mask_image)
    mask = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    # Find center
    x, y = cd.find_mask_center(mask_image)
    if x is not None:
        cd.draw_dot(res, x, y)

        # # Find 3D point
        # depth = 0.4
        # position = cd.pixel_to_3d_coordinate((x,y), depth, camera_matrix)
        # print(f"x{position[0]}\ny{position[1]}\n")

    # Show Image
    stacked = np.hstack((image, res))

    info = "[0-9]slots, [m]ove to, [q]uit"
    draw_text_box(
        image=stacked,
        text=info
    )
    
    slot_info = f"Color Slot [{cd.current_state_index}]"
    draw_text_box(
        image=stacked,
        text=slot_info,
        position="top_left"
    )
    scale = 0.8
    cv2.imshow(window,cv2.resize(stacked, None, fx=scale, fy=scale))

    # Input
    key = cv2.waitKey(1) & 0xFF
    key_str = chr(key)

    if key_str.isdigit() and int(key_str) >= 0 and int(key_str) <= 9:
        key_number = int(key_str)
        cd.current_state_index = key_number
        update_trackbars(window_name=window)
        print(f"Switching to {key_number}")

    elif key == ord('m'):
        print("Going to...")

    elif key == ord('q'):
        print("Bye :)")
        break

cv2.destroyAllWindows()
