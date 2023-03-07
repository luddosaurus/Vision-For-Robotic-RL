import cv2
from cv2 import aruco
import numpy as np
import pyrealsense2 as rs

# pipeline = rs.pipeline()
# config = rs.config()

# # Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))

# found_rgb = False
# for s in device.sensors:
#     if s.get_info(rs.camera_info.name) == 'RGB Camera':
#         found_rgb = True
#         break
# if not found_rgb:
#     print("The demo requires Depth camera with Color sensor")
#     exit(0)

# # Start streaming
# pipeline.start(config)

# try:
#     while True:

#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue

#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())


#         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#         depth_colormap_dim = depth_colormap.shape
#         color_colormap_dim = color_image.shape

#         # If depth and color resolutions are different, resize color image to match depth image for display
#         if depth_colormap_dim != color_colormap_dim:
#             resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
#             recolored_image = cv2.cvtColor(resized_color_image, cv2.COLOR_RGB2BGR)
#             images = np.hstack((recolored_image, depth_colormap))
#         else:
#             images = np.hstack((color_image, depth_colormap))

#         # Show images
#         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('RealSense', images)
#         cv2.waitKey(1)

# finally:

#     # Stop streaming
#     pipeline.stop()

def main():

    # vid = cv2.VideoCapture(0)

    # while True:
    #     ret, frame = vid.read()
    #     cv2.imshow('camera', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # vid.release()

    my_test = cv2.imread('./test.png')    
    my_test = cv2.resize(my_test, (0,0), fx=0.5, fy=0.5)
    # my_test = cv2.rotate(my_test, rotateCode=cv2.ROTATE_180)
    my_test_gray = cv2.cvtColor(my_test, cv2.COLOR_RGB2GRAY)
    my_test_something = cv2.cvtColor(my_test, cv2.COLOR_RGB2BGRA)

    test_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    for count in range(50):
        img_mark = aruco.generateImageMarker(test_aruco, count, 500)
        cv2.imshow('mark', img_mark)
        cv2.imwrite(f'./aruco_marks/mark_{count}.png', img_mark)
        cv2.waitKey(1)

    # my_test_gray = cv2.cvtColor(my_test, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('test_something', my_test_something)
    # cv2.imshow('test', my_test)
    cv2.imshow('test_gray', my_test_gray)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__== '__main__':
    main()
