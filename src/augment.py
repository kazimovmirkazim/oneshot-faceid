"""Data augmentation for the anchor and positive folders.

A few hundred webcam shots of one face are a small dataset. Each original image
is turned into N variants with small changes in brightness, contrast, mirror
image, JPEG quality and saturation, so the model sees more lighting and camera
conditions without collecting more photos.

Two details differ from the common tutorial version on purpose:
- every variant starts from the ORIGINAL image (no compounding of edits), and
- every transform gets a fresh random seed (a fixed seed repeats the same change).

Usage: python -m src.augment --n 9
"""
import argparse
import uuid

import numpy as np
import tensorflow as tf

from src import config

AUG_PREFIX = "aug_"


def augment(img: tf.Tensor, n: int = 9, rng: np.random.Generator | None = None) -> list[tf.Tensor]:
    rng = rng or np.random.default_rng()
    seed = lambda: tuple(int(x) for x in rng.integers(0, 2**31 - 1, size=2))  # noqa: E731
    out = []
    for _ in range(n):
        x = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=seed())
        x = tf.image.stateless_random_contrast(x, lower=0.6, upper=1.0, seed=seed())
        x = tf.image.stateless_random_flip_left_right(x, seed=seed())
        x = tf.image.stateless_random_jpeg_quality(x, min_jpeg_quality=90, max_jpeg_quality=100, seed=seed())
        x = tf.image.stateless_random_saturation(x, lower=0.9, upper=1.0, seed=seed())
        out.append(x)
    return out


def augment_folder(folder, n: int = 9, seed: int = 42) -> int:
    """Write n augmented copies of every original image in `folder`."""
    rng = np.random.default_rng(seed)
    written = 0
    originals = [p for p in sorted(folder.glob("*.jpg")) if not p.name.startswith(AUG_PREFIX)]
    for path in originals:
        img = tf.io.decode_jpeg(tf.io.read_file(str(path)), channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)          # [0, 1] for the colour ops
        for aug in augment(img, n, rng):
            out = tf.image.convert_image_dtype(tf.clip_by_value(aug, 0.0, 1.0), tf.uint8)
            target = str(folder / f"{AUG_PREFIX}{uuid.uuid1()}.jpg")
            tf.io.write_file(target, tf.io.encode_jpeg(out))
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9, help="augmented copies per original image")
    args = ap.parse_args()
    for folder in (config.ANC_PATH, config.POS_PATH):
        print(f"{folder.name}: wrote {augment_folder(folder, args.n)} augmented images")


if __name__ == "__main__":
    main()
