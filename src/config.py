"""Project paths. Everything is relative to the repository root, so the
code runs the same on Windows, macOS and Linux."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
POS_PATH = DATA_DIR / "positive"   # your face, second capture
ANC_PATH = DATA_DIR / "anchor"     # your face, reference capture
NEG_PATH = DATA_DIR / "negative"   # other people (LFW dataset)

APP_DATA = ROOT / "application_data"
INPUT_DIR = APP_DATA / "input_image"
VERIFICATION_DIR = APP_DATA / "verification_images"

CHECKPOINT_DIR = ROOT / "training_checkpoints"
MODEL_PATH = ROOT / "models" / "siamese_model.keras"

# Webcam crop: a 250x250 box around the face
CROP_Y, CROP_X, CROP_SIZE = 120, 200, 250

SAMPLES_PER_CLASS = 300
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.7
