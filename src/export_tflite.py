# src/export_tflite.py

import os
import tensorflow as tf
import tensorflow_hub as hub

from classifier_model import build_embedding_classifier

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


def build_serving_model(
    num_classes: int,
    classifier_weights_path: str,
) -> tf.keras.Model:

    # Raw audio input (1D, variable length)
    audio_input = tf.keras.layers.Input(
        shape=(),
        dtype=tf.float32,
        name="audio",
    )

    # YAMNet from TF Hub as a KerasLayer (non-trainable at serving time)
    yamnet_layer = hub.KerasLayer(
        YAMNET_HANDLE,
        trainable=False,
        name="yamnet",
    )

    scores, embeddings, spectrogram = yamnet_layer(audio_input)

    # Pool YAMNet embeddings over time
    pooled_embeddings = tf.reduce_mean(embeddings, axis=0, keepdims=True)  # [1, 1024]

    # Build the classifier and load trained weights
    classifier = build_embedding_classifier(num_classes=num_classes)
    classifier.build(input_shape=(None, 1024))
    classifier.load_weights(classifier_weights_path)

    logits = classifier(pooled_embeddings)

    serving_model = tf.keras.Model(
        inputs=audio_input,
        outputs=logits,
        name="yamnet_with_classifier",
    )

    return serving_model


def export_tflite(
    serving_model: tf.keras.Model,
    tflite_path: str,
    quantize: bool = False,
):
    converter = tf.lite.TFLiteConverter.from_keras_model(serving_model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {tflite_path}")


if __name__ == "__main__":
    num_classes = 10  # TODO: change this to your actual number of classes
    classifier_weights = os.path.join("models", "classifier_best.keras")
    tflite_output = os.path.join("models", "yamnet_classifier.tflite")

    if not os.path.exists(classifier_weights):
        raise FileNotFoundError(
            f"Could not find classifier weights at {classifier_weights}. "
            "Train the classifier first using train.py."
        )

    model = build_serving_model(
        num_classes=num_classes,
        classifier_weights_path=classifier_weights,
    )
    export_tflite(model, tflite_output, quantize=False)
