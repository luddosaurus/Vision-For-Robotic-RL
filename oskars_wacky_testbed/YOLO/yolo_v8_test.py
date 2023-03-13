import matplotlib
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

matplotlib.use("QT5Agg")


def fetch_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = np.asarray(image)
    return image


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)


def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    # Define COCO Labels
    if labels == []:
        labels = {0: u'__background__', 1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane',
                  6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant',
                  12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog',
                  18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe',
                  25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee',
                  31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat',
                  36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle',
                  41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana',
                  48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog',
                  54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed',
                  61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote',
                  67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink',
                  73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear',
                  79: u'hair drier', 80: u'toothbrush'}
    # Define colors
    if colors == []:
        # colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
        colors = [(89, 161, 197), (67, 161, 255), (19, 222, 24), (186, 55, 2), (167, 146, 11), (190, 76, 98),
                  (130, 172, 179), (115, 209, 128), (204, 79, 135), (136, 126, 185), (209, 213, 45), (44, 52, 10),
                  (101, 158, 121), (179, 124, 12), (25, 33, 189), (45, 115, 11), (73, 197, 184), (62, 225, 221),
                  (32, 46, 52), (20, 165, 16), (54, 15, 57), (12, 150, 9), (10, 46, 99), (94, 89, 46), (48, 37, 106),
                  (42, 10, 96), (7, 164, 128), (98, 213, 120), (40, 5, 219), (54, 25, 150), (251, 74, 172),
                  (0, 236, 196), (21, 104, 190), (226, 74, 232), (120, 67, 25), (191, 106, 197), (8, 15, 134),
                  (21, 2, 1), (142, 63, 109), (133, 148, 146), (187, 77, 253), (155, 22, 122), (218, 130, 77),
                  (164, 102, 79), (43, 152, 125), (185, 124, 151), (95, 159, 238), (128, 89, 85), (228, 6, 60),
                  (6, 41, 210), (11, 1, 133), (30, 96, 58), (230, 136, 109), (126, 45, 174), (164, 63, 165),
                  (32, 111, 29), (232, 40, 70), (55, 31, 198), (148, 211, 129), (10, 186, 211), (181, 201, 94),
                  (55, 35, 92), (129, 140, 233), (70, 250, 116), (61, 209, 152), (216, 21, 138), (100, 0, 176),
                  (3, 42, 70), (151, 13, 44), (216, 102, 88), (125, 216, 93), (171, 236, 47), (253, 127, 103),
                  (205, 137, 244), (193, 137, 224), (36, 152, 214), (17, 50, 238), (154, 165, 67), (114, 129, 60),
                  (119, 24, 48), (73, 8, 110)]

    # plot each boxes
    for box in boxes:
        # add score in label if score=True
        if score:
            label = labels[int(box[-1]) + 1] + " " + str(round(100 * float(box[-2]), 1)) + "%"
        else:
            label = labels[int(box[-1]) + 1]
        # filter every box under conf threshold if conf threshold setted
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)

    # show image

    return image


def get_camera_image(frame):
    image = np.asanyarray(frame.get_data())

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def main():
    # load model
    model = YOLO('yolov8s-seg.pt')
    # model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)
    # camera setup

    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # Start streaming
    pipeline.start(config)

    # url = "https://images.unsplash.com/photo-1600880292203-757bb62b4baf?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80"
    # image = fetch_image(url)

    # camera loop
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = get_camera_image(color_frame)
            results = model.predict(image)
            # print(results[0].masks.masks[0][0])
            if len(results) != 0:
                test = np.asanyarray(results[0].masks.masks[0], dtype=np.float64)
                # print(f'segments: {len(results[0].masks.segments)}')
                segment = np.asanyarray((results[0].masks.segments[0]))

                height, width, _ = image.shape
                black_image = np.zeros((height, width, 3), np.uint8)
                black_image[:] = (0, 0, 0)

                box = results[0].boxes.boxes[0]

                x = int(box[0])
                y = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                # print(f'{(box[0], box[1])}, {(box[2], box[3])}')

                roi = black_image[y: y2, x: x2]

                roi_height, roi_width, _ = roi.shape

                # print(roi.shape)
                # print(f'mask: {test.shape}')
                _, mask = cv2.threshold(test, 0.5, 255, cv2.THRESH_BINARY)
                # _, black_image = cv2.threshold(test, 0.5, 255, cv2.THRESH_BINARY)
                image = cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 3)
                contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print(contours)
                color = [0, 255, 0]
                # cnt1 = contours[0]
                # fig1 = plt.figure()
                # cnt1 = cnt1.reshape(contours[0].shape[0], contours[0].shape[2])
                # cnt_x = [x_i[0] for x_i in cnt1]
                # cnt_y = [x_i[1] for x_i in cnt1]
                # plt.plot(cnt_x, cnt_y)
                # plt.show()
                #

                # cv2.imshow('mask', mask)
                shape = segment.shape
                print(shape)
                for line in segment:
                    line[0] = line[0] * width
                    line[1] = line[1] * height
                    # print(line)
                # seg_x = [x_i[1] for x_i in segment]
                # seg_y = [x_i[0] for x_i in segment]
                # plt.plot(seg_x, seg_y)
                # plt.show()
                seg = segment.reshape((shape[0], 1, shape[1]))
                # print(f'seg: {seg}\ncnt: {contours[0]}')
                # print(f'cnt: {contours[0].shape}\nseg: {segment.shape}')

                cv2.fillPoly(black_image, np.int32([seg]),
                             (int(color[0]), int(color[1]), int(color[2])))
                # black_image = cv2.bitwise_not(black_image)
                # for cnt in contours:
                #     # print(cnt)
                #     cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))

                # print(mask)
                # image = plot_bboxes(image, results[0].boxes.boxes, score=True)

                # # test = np.zeros((480, 640, 3))
                # if test.shape == (3, 480, 640):
                #     test = np.reshape(test, (480, 640, 3))
                # print(test.max())
                b1 = 0.4 * black_image
                # i1 = 0.4 * image
                final_frame = cv2.bitwise_or(b1.astype("uint8"), image.astype("uint8"))
                # final_frame = ((0.6 * black_image) + (0.4 * image)).astype("uint8")
                cv2.imshow('black', black_image)
                cv2.imshow('final', final_frame)
            cv2.imshow('test', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
