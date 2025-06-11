import io
import numpy as np
from PIL import Image
import cv2

from rembg import remove


def remove_background(img_path):
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()

    output_image = remove(img_data)

    img_pil = Image.open(io.BytesIO(output_image)).convert("RGBA")
    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)

    return result