import cv2
import numpy as np


# Draw stuffs on images
class DaVinci:

    @staticmethod
    def draw_text_box_in_center(
            image,
            text,
            background=(255, 0, 255),
            foreground=(255, 255, 255),
            font_scale=1.0,
            thickness=2,
            margin=40):

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get the image dimensions
        height, width, _ = image.shape

        # Get the text size
        text_size_list = []
        for line in text:
            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_size_list.append(text_size)

        for i, line in enumerate(text):
            x = int(image.shape[1] / 2 - text_size_list[i][0] / 2)
            y = margin * (i + 1) + text_size_list[i][1]

            image = DaVinci.draw_text_box(image=image, text_size=text_size_list[i], x=x,
                                          y=y, text=line, foreground=foreground,
                                          background=(0, 0, 0), thickness=thickness, font_scale=font_scale)
        return image

    @staticmethod
    def draw_text_box_in_corner(
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

        # Get the text size
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Set the position coordinates based on the input position
        if position == 'top_left':
            x = margin
            y = margin
        elif position == 'top_right':
            x = width - text_size[0] - margin
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

        if box:
            image = DaVinci.draw_text_box(text=text, x=x, y=y, text_size=text_size, image=image, foreground=foreground,
                                          background=background, thickness=thickness, font_scale=font_scale)

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

    @staticmethod
    def draw_text_box(
            image,
            text,
            x,
            y,
            text_size,
            background=(255, 0, 255),
            foreground=(255, 255, 255),
            font_scale=1.0,
            thickness=2):
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

        return image

    @staticmethod
    def resize(image, target_width=1280):
        height, width = image.shape[:2]
        aspect_ratio = width / height
        target_height = int(target_width / aspect_ratio)
        resized_image = cv2.resize(image, (target_width, target_height))

        return resized_image

    @staticmethod
    def draw_charuco_corner(image, corner):
        cv2.circle(img=image, center=(int(corner[0][0]), int(corner[0][1])), radius=10, color=(255, 255, 0),
                   thickness=-1)
        return image

    @staticmethod
    def pad_image(image):
        # height, width = image.shape[:2]
        padded_image = np.pad(image, ((100, 100, 3), (0, 0, 0)), mode='constant')
        return padded_image

    @staticmethod
    def pad_image_cv(image, top_pad=200, bottom_pad=200, right_pad=200, left_pad=200, color=(0, 0, 0)):

        padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                          value=color)
        return padded_image
