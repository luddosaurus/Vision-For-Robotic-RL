from Calibrator import Calibrator
import cv2
import glob
import os

def main():
    chessboard_size = (5,8)
    chessboard_square_size = 20
    calibrator = Calibrator(chessboard_size, chessboard_square_size)

    calibartion_images = glob.glob('/home/oskarlarsson/Documents/Programs/Python_tests/real_sense_camera/images_real_sense/*.png')


    for calibration_image in calibartion_images:
        current_image = cv2.imread(calibration_image)
        chess_image = calibrator.update_chessboard_points(current_image)
        cv2.imshow('image', chess_image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            continue


    # while True:
    # # Run Camera
    #     image = camera.fetch()
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break

    #     # Do Stuff
    #     chess_image = calibrator.update_chessboard_points(image)

    #     # Show Image
    #     cv2.imshow("Video", chess_image)

    # Calibrate
    calibrated_image = calibrator.calibrate_intrinsic(img=current_image)
    cv2.imshow("Calibrated", calibrated_image)
    cv2.waitKey(0)
    print("Calibration pre-save ", calibrator.camera_matrix)

    # Save n Load calibration matrix
    save_path = "calibrations/matrix2"


    CHECK_DIR = os.path.isdir(save_path.split('/')[0])


    if not CHECK_DIR:
        os.makedirs(save_path.split('/')[0])
        print(f'"{save_path}" Directory is created')

    else:
        print(f'"{save_path}" Directory already Exists.')

    calibrator.save_calibration(path=save_path)

    calibrator.find_error()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()