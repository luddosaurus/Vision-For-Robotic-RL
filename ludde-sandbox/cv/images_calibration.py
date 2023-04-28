import cv2

from cv.utils.Calibrator import Calibrator
import glob
import random

path_images = "images/images1/*"

calibrator = Calibrator(size_y=5, size_x=8, square_size_mm=20)


def auto_calibrate(images, k_start=5, k_end=15, epochs=100):
    if len(images) == 0:
        return

    error_images = []

    for k in range(k_start, k_end):
        print("K = ", k)

        for e in range(epochs):
            c_images = random.choices(images, k=k)
            for calibration_image in c_images:
                current_image = cv2.imread(calibration_image)
                chess_image = calibrator.find_and_save_chessboard_points(current_image)

            calibrator.calibrate_intrinsic(img=current_image)
            error = calibrator.find_error()
            error_images.append((error, c_images))
            calibrator.clear_points()
            if epochs % 10 == 0:
                print("Epoch : ", e)

    # FInd min error
    print(min(error_images, key=lambda x: x[0]))
    print(max(error_images, key=lambda x: x[0]))


calibration_images = glob.glob(path_images)
auto_calibrate(calibration_images)




# for calibration_image in external_calibration_transforms:
#     current_image = cv2.imread(calibration_image)
#     chess_image = calibrator.find_and_save_chessboard_points(current_image)
#     # cv2.imshow('image', chess_image)
#     # if cv2.waitKey(20) & 0xFF == ord('q'):
#     #     continue

# Calibrate
# calibrated_image = calibrator.calibrate_intrinsic(img=current_image)
# cv2.imshow("Calibrated", calibrated_image)
# cv2.waitKey(0)
# print("Calibration:\n", calibrator.camera_matrix)
#
# # Save n Load calibration matrix
# save_name = "realsense"
# calibrator.save_calibration(save_name=save_name)
#
# calibrator.find_error()
#
# cv2.destroyAllWindows()
