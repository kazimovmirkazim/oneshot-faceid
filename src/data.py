"""Data collection helpers and the tf.data pipeline."""
import os
import shutil
import uuid
from pathlib import Path

import tensorflow as tf

from src import config
from src.model import IMG_SIZE


def make_dirs() -> None:
    for p in (config.POS_PATH, config.ANC_PATH, config.NEG_PATH,
              config.INPUT_DIR, config.VERIFICATION_DIR):
        p.mkdir(parents=True, exist_ok=True)


def move_lfw_to_negatives(lfw_root: str | Path) -> int:
    """Flatten the LFW folder structure (one folder per person) into data/negative."""
    lfw_root = Path(lfw_root)
    moved = 0
    for person_dir in lfw_root.iterdir():
        if not person_dir.is_dir():
            continue
        for img in person_dir.glob("*.jpg"):
            shutil.move(str(img), str(config.NEG_PATH / img.name))
            moved += 1
    return moved


def crop_frame(frame):
    y, x, s = config.CROP_Y, config.CROP_X, config.CROP_SIZE
    return frame[y:y + s, x:x + s]


def unique_name(folder: Path) -> str:
    return str(folder / f"{uuid.uuid1()}.jpg")


def preprocess(file_path):
    """Read a JPEG, resize to 100x100 and scale pixels to [0, 1]."""
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img / 255.0


def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


def build_datasets(n: int = config.SAMPLES_PER_CLASS):
    """(anchor, positive) pairs -> label 1, (anchor, negative) pairs -> label 0.

    The file-path datasets are zipped first and images are decoded once,
    inside preprocess_twin (decoding them twice was a bug in my first draft).
    """
    def files(folder):
        pattern = os.path.join(str(folder), "*.jpg")
        return tf.data.Dataset.list_files(pattern, shuffle=True).take(n)

    anchor, positive, negative = files(config.ANC_PATH), files(config.POS_PATH), files(config.NEG_PATH)

    counts = [len(list(d.as_numpy_iterator())) for d in (anchor, positive, negative)]
    n_pairs = min([n] + counts)
    if n_pairs == 0:
        raise FileNotFoundError("One of data/anchor, data/positive, data/negative is empty.")
    anchor, positive, negative = anchor.take(n_pairs), positive.take(n_pairs), negative.take(n_pairs)
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(n_pairs))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(n_pairs))))
    data = positives.concatenate(negatives)

    data = data.map(preprocess_twin, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.cache()
    data = data.shuffle(buffer_size=config.SHUFFLE_BUFFER, seed=42, reshuffle_each_iteration=False)

    total = 2 * n_pairs
    n_train = round(total * config.TRAIN_SPLIT)

    train = data.take(n_train).batch(config.BATCH_SIZE).prefetch(8)
    test = data.skip(n_train).take(total - n_train).batch(config.BATCH_SIZE).prefetch(8)
    return train, test
