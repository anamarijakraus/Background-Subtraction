import cv2
import numpy as np


def show_image_with_alpha_cv2(image_with_alpha):
    if image_with_alpha.shape[2] != 4:
        raise ValueError("Input image does not have an alpha channel.")

    b, g, r, alpha = cv2.split(image_with_alpha)

    background = np.ones_like(b) * 255
    background = cv2.merge((background, background, background))

    alpha = alpha.astype(float) / 255.0
    alpha = cv2.merge([alpha] * 3)

    foreground = cv2.merge((b, g, r)).astype(float)
    background = background.astype(float)

    blended = cv2.multiply(alpha, foreground)
    background_blended = cv2.multiply(1 - alpha, background)

    result = cv2.add(blended, background_blended).astype(np.uint8)

    return result