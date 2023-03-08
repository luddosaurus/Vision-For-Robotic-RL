import cv2

from base.utils.Calibrator import Calibrator
import glob
import random

path_images = "images/images1/*"

calibrator = Calibrator(size_y=5, size_x=8, square_size_mm=20)


def auto_calibrate(images, k_start=10, k_end=15, epochs=100):
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
    calibrator.clear_points()

    c_images = error_images[1]
    for calibration_image in c_images:
        current_image = cv2.imread(calibration_image)
        chess_image = calibrator.find_and_save_chessboard_points(current_image)

    calibrator.calibrate_intrinsic(img=current_image)
    calibrator.save_calibration("auto")


def manual_calibration(images):

    for calibration_image in images:
        current_image = cv2.imread(calibration_image)
        found_image, corners = calibrator.find_chessboard(current_image)
        chess_image = calibrator.draw_chessboard_on_image(img=current_image, corners=corners, found_pattern=found_image)
        cv2.imshow("Points", chess_image)
        cv2.waitKey(0)
        # todo y/n
        calibrator.save_chessboard_points(corners)

    calibrator.calibrate_intrinsic(current_image)


calibration_images = glob.glob(path_images)
# auto_calibrate(calibration_images)
manual_calibration(calibration_images)
