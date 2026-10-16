"""Train the Siamese network, evaluate it on the held-out set and save it.

Usage: python -m src.train [--epochs 50]
"""
import argparse

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

from src import config
from src.data import build_datasets
from src.model import make_siamese_model


def train(model, train_data, epochs: int):
    loss_fn = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(config.LEARNING_RATE)
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=model)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(config.CHECKPOINT_DIR / "ckpt")

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            x, y = batch[:2], batch[2]
            yhat = model(x, training=True)
            y = tf.reshape(y, tf.shape(yhat))
            loss = loss_fn(y, yhat)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, yhat, y

    history = []
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        progbar = tf.keras.utils.Progbar(len(train_data))
        r, p = Recall(), Precision()
        epoch_loss = []
        for idx, batch in enumerate(train_data):
            loss, yhat, y = train_step(batch)
            r.update_state(y, yhat)
            p.update_state(y, yhat)
            epoch_loss.append(float(loss))
            progbar.update(idx + 1)
        row = {"epoch": epoch,
               "loss": sum(epoch_loss) / len(epoch_loss),
               "recall": float(r.result()),
               "precision": float(p.result())}
        history.append(row)
        print(f"loss={row['loss']:.4f}  recall={row['recall']:.3f}  precision={row['precision']:.3f}")
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=prefix)
    return history


def evaluate(model, test_data, threshold: float = 0.5) -> dict:
    """Precision, recall and accuracy over the whole test set, not one batch."""
    r, p = Recall(thresholds=threshold), Precision(thresholds=threshold)
    acc = tf.keras.metrics.BinaryAccuracy(threshold=threshold)
    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = model.predict([test_input, test_val], verbose=0)
        y_true = y_true.reshape(yhat.shape)
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat)
        acc.update_state(y_true, yhat)
    return {"precision": float(p.result()), "recall": float(r.result()), "accuracy": float(acc.result())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = ap.parse_args()

    train_data, test_data = build_datasets()
    model = make_siamese_model()
    model.summary()

    train(model, train_data, args.epochs)
    metrics = evaluate(model, test_data)
    print("\nHeld-out test set:", {k: round(v, 3) for k, v in metrics.items()})

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
