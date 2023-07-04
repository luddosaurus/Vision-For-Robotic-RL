import cv2
import numpy as np
import DaVinci


class ObjectDetector:

    def __init__(self) -> None:
        pass

    def detect_objects(self, img, min_area=1000):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        _, thresh = cv2.threshold(
             src=gray, thresh=175, maxval=255, type=cv2.THRESH_BINARY)
        

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small objects based on area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered_contours.append(contour)
        
        objects = []
        
        # Iterate over the contours
        for contour in contours:
            # Compute the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)

            rect = cv2.minAreaRect(contour)
            
            # Extract the center and size of the rectangle
            center, size, angle = rect
            
            # Calculate the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Extract the translation and rotation from the matrix
            translation = rotation_matrix[:, 2]
            rotation_rad = -np.radians(angle)  # Convert angle to radians
            
            # Create a dictionary to store object information
            obj = {
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'contour': contour,
                'translation': translation,
                'rotation': rotation_rad
            }
            
            # Add the object to the list
            objects.append(obj)
        
        return objects

    def draw_contours_on_image(self, image, contours):
        # Create a copy of the original image
        image_with_contours = image.copy()
        
        # Draw contours on the image
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        
        return image_with_contours

    def draw_contours_with_boxes(self, image, contours):
        # Create a copy of the original image
        image_with_boxes = image.copy()

        # Iterate over each contour
        for contour in contours:
            # Find the minimum area bounding rectangle for the contour
            rect = cv2.minAreaRect(contour)
            
            # Get the box coordinates and angle
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            angle = rect[2]
            
            # Draw the rotated bounding box on the image
            cv2.drawContours(image_with_boxes, [box], 0, (0, 255, 0), 2)
        
        return image_with_boxes
    
    def draw_pose_vectors(self, image, objects):
        for obj in objects:
            # Get the object's pose information
            translation = obj['translation']
            rotation = obj['rotation']
            
            # Calculate the endpoint of the pose vector
            endpoint = translation + rotation  # Adjust the calculation based on your pose representation
            
            # Convert the translation and endpoint to integer coordinates
            translation = tuple(map(int, translation))
            endpoint = tuple(map(int, endpoint))
            
            # Draw the pose vector on the image
            cv2.arrowedLine(image, translation, endpoint, (0, 255, 0), 2)  # Adjust the color and line thickness as desired
        
        return image

  


# ------------------------------ Main
window = 'ObjectDetection'
cv2.namedWindow(window)

cap = cv2.VideoCapture(0)
detector = ObjectDetector()

pose_estimate = True
if pose_estimate:
    with np.load("cv/intrinsic_matrix.npz") as X:
                camera_matrix, distortion_coefficients, _, _ = \
                    [X[i] for i in ('matrix', 'distortion', 'rotation_vectors', 'translation_vectors')]

depth = 0.4
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]

while True:
    # Get Image
    ret, image = cap.read()
    if not ret:
        break

    objects = detector.detect_objects(image)
    image_with_contours = detector.draw_contours_on_image(image, [obj['contour'] for obj in objects])
    image_with_box = detector.draw_contours_with_boxes(image, [obj['contour'] for obj in objects])
    pose = detector.draw_pose_vectors(image, objects)

    # Show Image
    stacked = np.hstack((image, image_with_contours, image_with_box, pose))
    scale = 0.5
    cv2.imshow(window, cv2.resize(stacked, None, fx=scale, fy=scale))

    # Input
    key = cv2.waitKey(1) & 0xFF
    key_str = chr(key)

    if key == ord('q'):
        print("Bye :)")
        break

cv2.destroyAllWindows()
