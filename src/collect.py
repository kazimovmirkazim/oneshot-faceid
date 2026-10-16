"""Collect anchor and positive images from the webcam.

Keys:  a = save anchor   p = save positive   q = quit
Usage: python -m src.collect
"""
import cv2

from src import config
from src.data import crop_frame, make_dirs, unique_name


def main(camera_index: int = 0) -> None:
    make_dirs()
    cap = cv2.VideoCapture(camera_index)
    counts = {"anchor": 0, "positive": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera frame could not be read - check the camera connection or index.")
            break

        cropped = crop_frame(frame)
        key = cv2.waitKey(1) & 0xFF   # read the key once per frame

        if key == ord("a"):
            cv2.imwrite(unique_name(config.ANC_PATH), cropped)
            counts["anchor"] += 1
        elif key == ord("p"):
            cv2.imwrite(unique_name(config.POS_PATH), cropped)
            counts["positive"] += 1
        elif key == ord("q"):
            break

        cv2.imshow("Image collection (a = anchor, p = positive, q = quit)", cropped)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {counts['anchor']} anchor and {counts['positive']} positive images.")


if __name__ == "__main__":
    main()
