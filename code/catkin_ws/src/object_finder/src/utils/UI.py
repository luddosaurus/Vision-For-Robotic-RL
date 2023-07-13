import cv2

from utils.Const import Const
from utils.DaVinci import DaVinci


class UI:

    def __init__(self):
        self.window = 'ColorDetection'
        self.gui_created = False

        self.display_image = None
        self.mouse_hover_x = None
        self.mouse_hover_y = None

    def create_layout(self, start_state, update_value, on_click):
        cv2.namedWindow(self.window)

        cv2.createTrackbar("Hue", self.window,
                           start_state[Const.HUE], Const.HUE_MAX,
                           lambda value: update_value(value, Const.HUE))
        cv2.createTrackbar("Saturation", self.window,
                           start_state[Const.SATURATION], Const.SAT_MAX,
                           lambda value: update_value(value, Const.SATURATION))
        cv2.createTrackbar("Value", self.window,
                           start_state[Const.VALUE], Const.VAL_MAX,
                           lambda value: update_value(value, Const.VALUE))

        cv2.createTrackbar("Hue Margin", self.window,
                           start_state[Const.HUE_MARGIN], Const.HUE_MAX,
                           lambda value: update_value(value, Const.HUE_MARGIN))
        cv2.createTrackbar("Sat Margin", self.window,
                           start_state[Const.SATURATION_MARGIN], Const.SAT_MAX,
                           lambda value: update_value(value, Const.SATURATION_MARGIN))
        cv2.createTrackbar("Val Margin", self.window,
                           start_state[Const.VALUE_MARGIN], Const.VAL_MAX,
                           lambda value: update_value(value, Const.VALUE_MARGIN))

        cv2.createTrackbar("Noise", self.window,
                           start_state[Const.NOISE], Const.NOISE_MAX,
                           lambda value: update_value(value, Const.NOISE))
        cv2.createTrackbar("Fill", self.window,
                           start_state[Const.FILL], Const.FILL_MAX,
                           lambda value: update_value(value, Const.FILL))

        cv2.setMouseCallback(self.window, on_click)
        self.gui_created = True

    def update_trackbars(self, current_state):
        cv2.setTrackbarPos("Hue", self.window, current_state[Const.HUE])
        cv2.setTrackbarPos("Saturation", self.window, current_state[Const.SATURATION])
        cv2.setTrackbarPos("Value", self.window, current_state[Const.VALUE])
        cv2.setTrackbarPos("Hue Margin", self.window, current_state[Const.HUE_MARGIN])
        cv2.setTrackbarPos("Sat Margin", self.window, current_state[Const.SATURATION_MARGIN])
        cv2.setTrackbarPos("Val Margin", self.window, current_state[Const.VALUE_MARGIN])
        cv2.setTrackbarPos("Noise", self.window, current_state[Const.NOISE])
        cv2.setTrackbarPos("Fill", self.window, current_state[Const.FILL])

    def update_mouse_hover(self, x, y):
        self.mouse_hover_x = x
        self.mouse_hover_y = y

    def update_ui(self, current_state_index, scale, roi_size):
        info = "[0-9] states, [m]ove to, [q]uit"
        DaVinci.draw_text_box(
            image=self.display_image,
            text=info
        )

        slot_info = f"Color State [{current_state_index}]"
        DaVinci.draw_text_box(
            image=self.display_image,
            text=slot_info,
            position="top_left"
        )

        if self.mouse_hover_x is not None:
            self.display_image = DaVinci.draw_roi_rectangle(
                image=self.display_image,
                x=int(self.mouse_hover_x / scale),
                y=int(self.mouse_hover_y / scale),
                roi=roi_size
            )

        cv2.imshow(
            self.window,
            cv2.resize(self.display_image, None, fx=scale, fy=scale)
        )