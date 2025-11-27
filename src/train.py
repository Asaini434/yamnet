# src/train.py


# Right now this file is a working template with a placeholder dataset loader.


import os
from typing import Tuple

import tensorflow as tf

from yamnet_backbone import map_to_embedding
from classifier_model import build_embedding_classifier


# =========================
# 1. Dataset loading
# =========================

def load_waveform_dataset(
    data_root: str,
    split: str = "train",
) -> tf.data.Dataset:
    """
    (Tony): Implement real dataset loading.
    For now we raise NotImplementedError.
    """
    raise NotImplementedError(
        "Dataset loader not implemented yet. "
        "Tony: please implement load_waveform_dataset() in train.py."
    )


def make_embedding_dataset(
    wave_ds: tf.data.Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:

    ds = wave_ds.map(
        map_to_embedding,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        ds = ds.shuffle(10_000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =========================
# 2. Training loop
# =========================

def train(
    data_root: str,
    num_classes: int,
    batch_size: int = 64,
    epochs: int = 10,
    model_dir: str = "models",
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:

    # --- Load raw waveform datasets ---
    train_wave_ds = load_waveform_dataset(data_root, split="train")
    val_wave_ds = load_waveform_dataset(data_root, split="val")

    # --- Turn waveforms into embeddings via YAMNet ---
    train_ds = make_embedding_dataset(train_wave_ds, batch_size=batch_size, shuffle=True)
    val_ds = make_embedding_dataset(val_wave_ds, batch_size=batch_size, shuffle=False)

    # --- Build classifier on top of embeddings ---
    model = build_embedding_classifier(num_classes=num_classes)

    # Standard loss for now;
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=metrics,
    )

    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "classifier_best.keras"),
            monitor="val_acc",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "logs"),
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save final classifier
    final_path = os.path.join(model_dir, "classifier_final.keras")
    model.save(final_path)
    print(f"Saved final classifier to: {final_path}")

    return model, history


if __name__ == "__main__":
    # For now, running this file directly will raise NotImplementedError.
    pass
