import cv2

video = "res/test_video.mov"
cam = 0


def resize_and_crop_image(image, width, height):
    current_height, current_width = image.shape[:2]

    if current_width == width and current_height == height:
        # Image already has the desired size, return it without modification
        return image

    # Calculate the aspect ratio of the image
    aspect_ratio = current_width / current_height

    # Calculate the target aspect ratio
    target_aspect_ratio = width / height

    if aspect_ratio > target_aspect_ratio:
        # Resize the image based on width
        new_width = int(height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, height))
        # Calculate the crop region
        start = new_width - width
        cropped_image = resized_image[:, :-start]
    else:
        # Resize the image based on height
        new_height = int(width / aspect_ratio)
        resized_image = cv2.resize(image, (width, new_height))
        # Calculate the crop region
        start = new_height - height
        cropped_image = resized_image[:-start, :]

    return cropped_image


# Change source with 'video' / 'cam'
cap = cv2.VideoCapture(cam)

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    cv2.imshow("Resize", resize_and_crop_image(img, 1780, 720))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

