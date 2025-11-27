# src/classifier_model.py

import tensorflow as tf


def build_embedding_classifier(num_classes: int) -> tf.keras.Model:
    """
    Build a classifier for YAMNet embeddings.

    Input shape:  (1024,)
    Output shape: (num_classes,) logits
    """
    inputs = tf.keras.layers.Input(shape=(1024,), name="embedding")

    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="yamnet_embedding_classifier")
    return model
