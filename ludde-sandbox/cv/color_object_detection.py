import cv2
import numpy as np


class ColorObjectDetector:
    # HSV = Hue, Saturation, Value
    hue_max = 179
    sat_max = 255
    value_max = 255

    def __init__(self) -> None:
        pass

    def code2hsv(self, hue, sat, val):
        return [
        int(self.hue_max*(hue/360)),
        int(self.sat_max*sat),
        int(self.value_max*val),
    ]

    def get_hsv_mask(self, image, hsv, margin=20):

        lower_range = np.array(hsv) - margin
        upper_range = np.array(hsv) + margin


        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a mask using the specified lower and upper ranges
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        
        return mask
    
    def find_mask_center(self, mask):
        try:
            moments = cv2.moments(mask)

            # Calculate the centroid coordinates
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            return cx, cy
        
        except:
            return None, None
    
    def draw_dot(self, image, x, y, color=(255, 0, 255), radius=20, thickness=3):
        cv2.circle(image, (x, y), radius, color, thickness)

    def draw_text(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 255), thickness=1):
        cv2.putText(image, text, (20,20), font, font_scale, color, thickness)

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


# ------------------------------ Main
window = 'ColorDetection'
cv2.namedWindow(window)

cap = cv2.VideoCapture(0)
cd = ColorObjectDetector()

# todo find target with clicking

with np.load("cv/intrinsic_matrix.npz") as X:
            camera_matrix, distortion_coefficients, _, _ = \
                [X[i] for i in ('matrix', 'distortion', 'rotation_vectors', 'translation_vectors')]

while True:
    # Get Image
    ret, image = cap.read()
    if not ret:
        break
    
    target_color = cd.code2hsv(209, 0.88, 0.9)

    # Mask
    mask_image = cd.get_hsv_mask(image=image, hsv=target_color, margin=20)
    res = cv2.bitwise_and(image, image, mask=mask_image)
    mask = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    # Find center
    x, y = cd.find_mask_center(mask_image)
    if x is not None:
        cd.draw_dot(res, x, y)

        # Find 3D point
        depth = 0.4
        position = cd.pixel_to_3d_coordinate((x,y), depth, camera_matrix)
        print(f"x{position[0]}\ny{position[1]}\n")

    # Show Image
    stacked = np.hstack((res,image,mask))
    cv2.imshow(window,cv2.resize(stacked,None,fx=0.4,fy=0.4))


    # Exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
