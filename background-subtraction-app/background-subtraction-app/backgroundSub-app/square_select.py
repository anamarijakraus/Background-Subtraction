import cv2
import numpy as np

rect = (0, 0, 1, 1)
drawing = False
ix, iy = -1, -1
mask = None
display_img = None
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


def rem_bg_selection(img_new):
    img = cv2.imread(img_new)
    display_img = img.copy()

    def select_subject(event, x, y, flags, param):
        nonlocal display_img
        global ix, iy, rect, drawing, mask

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            mask = np.zeros(img.shape[:2], np.uint8)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                display_img = img.copy()
                cv2.rectangle(display_img, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Image', display_img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
            cv2.rectangle(display_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', display_img)

            x1, y1, x2, y2 = rect
            mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
            mask[mask == 0] = cv2.GC_BGD

            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_subject)

    print("Draw a rectangle around the subject. Press any key when done.")
    cv2.imshow('Image', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    edges = cv2.Canny(img, 300, 400)
    mask2[edges != 0] = 1

    b, g, r = cv2.split(img)
    alpha = mask2 * 255

    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    result = cv2.merge((b, g, r, alpha))

    return result