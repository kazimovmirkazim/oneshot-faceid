"""Desktop Face ID app (Kivy).

Run from the repository root:  python -m app.faceid
Needs a trained model in models/ and photos in application_data/verification_images.
"""
import os
import sys

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.data import crop_frame, make_dirs, preprocess  # noqa: E402
from src.model import load_siamese_model  # noqa: E402


class CamApp(App):
    detection_threshold = 0.5
    verification_threshold = 0.5

    def build(self):
        make_dirs()
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, 0.1))

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.model = load_siamese_model(str(config.MODEL_PATH))
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = crop_frame(frame)
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.web_cam.texture = texture

    def verify(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            self.verification_label.text = "Camera error"
            return
        cv2.imwrite(str(config.INPUT_DIR / "input_image.jpg"), crop_frame(frame))

        input_img = preprocess(str(config.INPUT_DIR / "input_image.jpg")).numpy()
        files = sorted(f for f in os.listdir(config.VERIFICATION_DIR) if f.lower().endswith(".jpg"))
        validation = np.stack([preprocess(str(config.VERIFICATION_DIR / f)).numpy() for f in files])
        inputs = np.repeat(input_img[None, ...], len(files), axis=0)
        results = self.model.predict([inputs, validation], verbose=0).ravel()

        detection = int(np.sum(results > self.detection_threshold))
        verified = detection / len(files) > self.verification_threshold
        self.verification_label.text = "Verified" if verified else "Unverified"

        Logger.info(f"Matching pairs: {detection}/{len(files)}  verified={verified}")
        return results, verified


if __name__ == "__main__":
    CamApp().run()
