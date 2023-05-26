import cv2


def draw_text_box(
        image,
        text,
        position="bottom_left",
        background=(255, 0, 255),
        foreground=(255, 255, 255),
        box=True):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    margin = 30
    # Get the image dimensions
    height, width, _ = image.shape

    # Set the position coordinates based on the input position
    if position == 'top_left':
        x = margin
        y = margin + width
    elif position == 'top_right':
        x = width - margin - height
        y = margin + width
    elif position == 'bottom_left':
        x = margin
        y = height - margin
    elif position == 'bottom_right':
        x = width - margin - height
        y = height - margin

    # Get the text size
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    if box:
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


# Load the image
image = cv2.imread('test.png')

# Draw the text box on the image
image_with_text_box = draw_text_box(
    image=image,
    text='Hello, World!',
    position='bottom_left',
    foreground=(0, 255, 255),
    background=(255, 0, 255))

# Display the image
cv2.imshow('Image with Text Box', image_with_text_box)
cv2.waitKey(0)
cv2.destroyAllWindows()
