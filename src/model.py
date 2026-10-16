"""Siamese network for one-shot face verification.

Architecture follows Koch, Zemel & Salakhutdinov (2015),
"Siamese Neural Networks for One-shot Image Recognition":
a shared CNN turns each image into a 4096-d embedding, an L1 distance
layer compares the two embeddings, and a sigmoid unit outputs the
probability that both images show the same person.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Layer, MaxPooling2D
from tensorflow.keras.models import Model

IMG_SIZE = 100


class L1Dist(Layer):
    """Element-wise absolute difference between two embeddings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        # Keras 3 may hand over a one-element list instead of a tensor
        if isinstance(input_embedding, (list, tuple)):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, (list, tuple)):
            validation_embedding = validation_embedding[0]
        return tf.math.abs(input_embedding - validation_embedding)


def make_embedding() -> Model:
    """Shared CNN: 100x100x3 image -> 4096-d embedding (~39M parameters)."""
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # Block 1
    c1 = Conv2D(64, (10, 10), activation="relu")(inp)
    m1 = MaxPooling2D(pool_size=(2, 2), padding="same")(c1)

    # Block 2
    c2 = Conv2D(128, (7, 7), activation="relu")(m1)
    m2 = MaxPooling2D(pool_size=(2, 2), padding="same")(c2)

    # Block 3
    c3 = Conv2D(128, (4, 4), activation="relu")(m2)
    m3 = MaxPooling2D(pool_size=(2, 2), padding="same")(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation="sigmoid")(f1)

    return Model(inputs=[inp], outputs=[d1], name="embedding")


def make_siamese_model(embedding: Model | None = None) -> Model:
    """Two inputs -> shared embedding -> L1 distance -> similarity score."""
    embedding = embedding or make_embedding()

    input_image = Input(name="input_img", shape=(IMG_SIZE, IMG_SIZE, 3))
    validation_image = Input(name="validation_img", shape=(IMG_SIZE, IMG_SIZE, 3))

    distances = L1Dist(name="distance")(embedding(input_image), embedding(validation_image))
    classifier = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name="SiameseNetwork")


def load_siamese_model(path: str) -> Model:
    """Load a saved model with the custom distance layer registered."""
    return tf.keras.models.load_model(
        path,
        custom_objects={"L1Dist": L1Dist, "BinaryCrossentropy": tf.losses.BinaryCrossentropy},
        compile=False,
    )
