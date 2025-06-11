import threading
import cv2
import numpy as np

brush_size = 10
drawing = False
mask = None
display_img = None
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
img = None
result = None

def select_subject(event, x, y, flags, param):
    global drawing, mask, display_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), brush_size, cv2.GC_PR_FGD, -1)
        display_img = img.copy()
        display_img[mask == cv2.GC_PR_FGD] = [0, 255, 0]
        cv2.imshow('Image', display_img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), brush_size, cv2.GC_PR_FGD, -1)
        display_img = img.copy()
        display_img[mask == cv2.GC_PR_FGD] = [0, 255, 0]
        cv2.imshow('Image', display_img)

def update_brush_size(val):
    global brush_size
    brush_size = max(1, val)

def _run_grabcut(img_new):
    global img, mask, display_img, result

    img = cv2.imread(img_new)
    display_img = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_subject)

    cv2.createTrackbar('Brush Size', 'Image', brush_size, 50, update_brush_size)

    print("Brush on the subject. Press 'q' when you're ready.")
    while True:
        cv2.imshow('Image', display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    mask[mask == 0] = cv2.GC_BGD
    mask[mask == cv2.GC_PR_FGD] = cv2.GC_PR_FGD

    try:
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        print("OpenCV error:", e)
        cv2.destroyAllWindows()
        return None

    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    mask2 = cv2.dilate(mask2, np.ones((10, 10), np.uint8), iterations=1)
    mask2 = cv2.erode(mask2, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.Canny(img, 200, 200)
    mask2[edges != 0] = 1
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask2, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 500:
            mask2[labels == i] = 0

    b, g, r = cv2.split(img)
    alpha = cv2.GaussianBlur(mask2 * 255, (3, 3), 0)
    result = cv2.merge((b, g, r, alpha))

    cv2.destroyAllWindows()

def background_subtraction_method_3(img_new):
    thread = threading.Thread(target=_run_grabcut, args=(img_new,))
    thread.start()
    thread.join()
    return result