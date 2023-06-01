import cv2
import numpy as np

def calculate_reprojection_error(image, camera_matrix, distortion_coeffs, object_points):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners (or the pattern of your calibration target)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        # Refine the corner locations
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Project the object points to image points
        _, rvecs, tvecs = cv2.solvePnP(object_points, corners, camera_matrix, distortion_coeffs)
        image_points, _ = cv2.projectPoints(object_points, rvecs, tvecs, camera_matrix, distortion_coeffs)

        # Calculate the reprojection error for each pixel
        error = np.abs(corners - image_points.squeeze())

        return error

    else:
        print("Chessboard corners not found.")
        return None

def compute_reprojection_error_for_image(image, cameraMatrix, distCoeffs, rvecs, tvecs):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the image width and height
    height, width = gray.shape[:2]

    # Initialize an empty array to store the re-projection errors for each pixel
    reprojection_errors = np.zeros((height, width))

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Compute the re-projection error for the current pixel coordinate
            pixel_coord = (x, y)
            reprojection_error = compute_reprojection_error(pixel_coord, cameraMatrix, distCoeffs, rvecs, tvecs)

            # Store the re-projection error in the array
            reprojection_errors[y, x] = reprojection_error

    return reprojection_errors


with np.load("cv/intrinsic_matrix.npz") as X:
            camera_matrix, distortion_coefficients, r_vecs, t_vecs = \
                [X[i] for i in ('matrix', 'distortion', 'rotation_vectors', 'translation_vectors')]
            
pixel_coord = (20, 20)
reproj_error = compute_reprojection_error(pixel_coord, camera_matrix, distortion_coefficients, r_vecs, t_vecs)
print("Re-projection Error:", reproj_error)