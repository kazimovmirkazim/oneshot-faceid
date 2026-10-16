"""Real-time face verification with OpenCV.

Put 30-50 photos of yourself in application_data/verification_images,
then run: python -m src.verify
Press v to verify the current frame, q to quit.
"""
import os

import cv2
import numpy as np

from src import config
from src.data import crop_frame, make_dirs, preprocess
from src.model import load_siamese_model


def verify(model, detection_threshold: float = 0.5, verification_threshold: float = 0.5):
    """Compare the input image with every verification image.

    detection_threshold:    score above which one pair counts as a match
    verification_threshold: share of matching pairs needed to verify the person
    """
    input_img = preprocess(str(config.INPUT_DIR / "input_image.jpg"))
    files = sorted(f for f in os.listdir(config.VERIFICATION_DIR) if f.lower().endswith(".jpg"))
    if not files:
        raise FileNotFoundError(f"No .jpg files in {config.VERIFICATION_DIR}")

    validation = np.stack([preprocess(str(config.VERIFICATION_DIR / f)).numpy() for f in files])
    inputs = np.repeat(np.expand_dims(input_img.numpy(), 0), len(files), axis=0)
    results = model.predict([inputs, validation], verbose=0).ravel()

    detection = int(np.sum(results > detection_threshold))
    verification = detection / len(files)
    return results, verification > verification_threshold


def main(camera_index: int = 0):
    make_dirs()
    model = load_siamese_model(str(config.MODEL_PATH))
    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = crop_frame(frame)
        cv2.imshow("Verification (v = verify, q = quit)", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("v"):
            cv2.imwrite(str(config.INPUT_DIR / "input_image.jpg"), frame)
            results, verified = verify(model, 0.5, 0.5)
            print(f"Verified: {verified}  |  matching pairs: {int((results > 0.5).sum())}/{len(results)}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
