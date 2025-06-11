import brush as br
import subtract as rem
import square_select as square
import camera as cam
import cv2
import numpy as np


def automatic_subtraction(img):
    return rem.remove_background(img)

def square_select_subtraction(img):
    return square.rem_bg_selection(img)

def brush_subtraction(img):
    return br.background_subtraction_method_3(img)

def camera_subtraction(img):
    return cam.cam_bg(img)

def replace_background(foreground_img, background_img):
    if foreground_img.shape[2] != 4:
        raise ValueError("Foreground image must have an alpha channel")

    h_fg, w_fg = foreground_img.shape[:2]
    background_resized = cv2.resize(background_img, (w_fg, h_fg))

    b, g, r, a = cv2.split(foreground_img)
    alpha = a.astype(float) / 255.0
    alpha = cv2.merge([alpha] * 3)

    foreground_rgb = cv2.merge([b, g, r]).astype(float)
    background_rgb = background_resized.astype(float)

    blended = cv2.multiply(alpha, foreground_rgb) + cv2.multiply(1 - alpha, background_rgb)
    blended = blended.astype(np.uint8)

    return blended