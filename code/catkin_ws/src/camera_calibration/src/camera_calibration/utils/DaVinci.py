import cv2


# Draw stuffs on images
class DaVinci:

    @staticmethod
    def draw_text(image, text, position='top_right', color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        margin = 10
        x = 0
        y = 0

        height, width, channels = image.shape
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        if position == 'top_left':
            x = margin
            y = margin + text_size[1]
        elif position == 'top_right':
            x = width - margin - text_size[0]
            y = margin + text_size[1]
        elif position == 'bottom_left':
            x = margin
            y = height - margin
        elif position == 'bottom_right':
            x = width - margin - text_size[0]
            y = height - margin

        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

        return image

    @staticmethod
    def resize(image, target_width=1280):
        height, width = image.shape[:2]
        aspect_ratio = width / height
        target_height = int(target_width / aspect_ratio)
        resized_image = cv2.resize(image, (target_width, target_height))

        return resized_image

