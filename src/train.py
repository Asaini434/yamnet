# src/train.py


# Right now this file is a working template with a placeholder dataset loader.


import os
from typing import Tuple, List, Optional

import tensorflow as tf

from yamnet_backbone import map_to_embedding
from classifier_model import build_embedding_classifier

CLASS_NAMES: Optional[List[str]] = None

# =========================
# 1. Dataset loading
# =========================

def load_waveform_dataset(
    data_root: str,
    split: str = "train",
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Loads GTZAN-based dataset from path (datasets/gtzan/[split]/[genre]/*.wav)
        [split]: train, val, test
    Returns:
        tf.data.Dataset of (waveform, label_index)
        -waveform: 1-D float32 tensor at 16 kHz
        -label_index: scalar int64
    """
    import pathlib
    global CLASS_NAMES

    split_dir = pathlib.Path(data_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories under {split_dir}")
    if CLASS_NAMES is None:
        CLASS_NAMES = [d.name for d in class_dirs]
        print("Discovered classes:", CLASS_NAMES)

    filepaths = []
    labels = []
    for class_idx, class_dir in enumerate(class_dirs):
        for wav_path in class_dir.glob("*.wav"):
            filepaths.append(str(wav_path))
            labels.append(class_idx)
    if not filepaths:
        raise RuntimeError(f"No .wav files found under {split_dir}")
    path_ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _load_wav(path, label):
        audio_bin = tf.io.read_file(path)
        waveform, sample_rate = tf.audio.decode_wav(audio_bin)
        waveform = tf.reduce_mean(waveform, axis=-1)  # (num_samples,)
        waveform = tf.cast(waveform, tf.float32)
        sample_rate = tf.cast(sample_rate, tf.int32)
        target_sr = 16000
        def _resample():
            num_samples = tf.shape(waveform)[0]
            ratio = tf.cast(target_sr, tf.float32) / tf.cast(sample_rate, tf.float32)
            new_len = tf.cast(tf.cast(num_samples, tf.float32) * ratio, tf.int32)
            wav_2d = tf.reshape(waveform, [1, -1, 1])
            wav_resized = tf.image.resize(wav_2d, [1, new_len], method="bilinear")
            wav_out = tf.reshape(wav_resized, [-1])
            return wav_out
        waveform_16k = tf.cond(
            sample_rate == target_sr,
            lambda: waveform,
            _resample,
        )
        return waveform_16k, tf.cast(label, tf.int64)

    ds = path_ds.map(_load_wav, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

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
