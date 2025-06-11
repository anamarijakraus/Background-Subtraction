import cv2
import mediapipe as mp
import numpy as np

def cam_bg(path):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    background_image_path = path
    background_image = cv2.imread(background_image_path)

    def resize_background(frame, background):
        return cv2.resize(background, (frame.shape[1], frame.shape[0]))

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask

        condition = mask > 0.5

        resized_background = resize_background(frame, background_image)

        output_image = np.where(condition[:, :, None], frame, resized_background)

        cv2.imshow('Background Removal and Replacement', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
