# trainAugment.py
import os
from typing import Tuple, List, Optional

import tensorflow as tf

from yamnet_backbone import map_to_embedding
from classifier_model import build_embedding_classifier

CLASS_NAMES: Optional[List[str]] = None


# 1. Dataset loading
def load_waveform_dataset(
    data_root: str,
    split: str = "train",
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Loads GTZAN dataset from datasets/gtzan/[split]/[genre]/*.wav"""
    import pathlib
    global CLASS_NAMES

    split_dir = pathlib.Path(data_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if CLASS_NAMES is None:
        CLASS_NAMES = [d.name for d in class_dirs]
        print("Discovered classes:", CLASS_NAMES)

    filepaths, labels = [], []
    for class_idx, class_dir in enumerate(class_dirs):
        for wav_path in class_dir.glob("*.wav"):
            filepaths.append(str(wav_path))
            labels.append(class_idx)

    path_ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _load_wav(path, label):
        audio_bin = tf.io.read_file(path)
        waveform, sample_rate = tf.audio.decode_wav(audio_bin)
        waveform = tf.reduce_mean(waveform, axis=-1)
        waveform = tf.cast(waveform, tf.float32)
        target_sr = 16000
        sample_rate = tf.cast(sample_rate, tf.int32)

        def _resample():
            num_samples = tf.shape(waveform)[0]
            ratio = tf.cast(target_sr, tf.float32) / tf.cast(sample_rate, tf.float32)
            new_len = tf.cast(tf.cast(num_samples, tf.float32) * ratio, tf.int32)
            wav_2d = tf.reshape(waveform, [1, -1, 1])
            wav_resized = tf.image.resize(wav_2d, [1, new_len], method="bilinear")
            return tf.reshape(wav_resized, [-1])

        waveform_16k = tf.cond(sample_rate == target_sr, lambda: waveform, _resample)
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
    ds = wave_ds.map(map_to_embedding, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10_000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# 2. Data augmentation
def apply_random_distortion(waveform, label,
                            noise_level=0.05,
                            pitch_range=4,
                            time_stretch_range=(0.8, 1.2)):
    """Randomly apply noise, pitch shift, or time stretch"""
    # Noise
    if tf.random.uniform([]) < 0.5:
        rms = tf.sqrt(tf.reduce_mean(tf.square(waveform)) + 1e-8)
        noise = tf.random.normal(tf.shape(waveform), stddev=rms * noise_level)
        waveform = tf.clip_by_value(waveform + noise, -1.0, 1.0)

    # Time stretch
    if tf.random.uniform([]) < 0.5:
        rate = tf.random.uniform([], *time_stretch_range)
        orig_len = tf.shape(waveform)[0]
        new_len = tf.cast(tf.cast(orig_len, tf.float32) / rate, tf.int32)
        waveform_2d = tf.expand_dims(waveform, axis=0)[..., tf.newaxis]
        stretched = tf.image.resize(waveform_2d, [1, new_len], method="bilinear")
        waveform = tf.squeeze(stretched, axis=[0, -1])

    # Pitch shift
    if tf.random.uniform([]) < 0.5:
        n_steps = tf.random.uniform([], -pitch_range, pitch_range, dtype=tf.int32)
        factor = 2 ** (tf.cast(n_steps, tf.float32) / 12.0)
        orig_len = tf.shape(waveform)[0]
        new_len = tf.cast(tf.cast(orig_len, tf.float32) / factor, tf.int32)
        waveform_2d = tf.expand_dims(waveform, axis=0)[..., tf.newaxis]
        shifted = tf.image.resize(waveform_2d, [1, new_len], method="bilinear")
        waveform = tf.squeeze(shifted, axis=[0, -1])

    return waveform, label


def augment_dataset(ds: tf.data.Dataset, fraction: float = 0.5) -> tf.data.Dataset:
    """Return dataset containing all original samples plus fraction of augmented samples"""
    # Original dataset
    orig_ds = ds

    # Fraction of augmented dataset
    aug_ds = ds.map(apply_random_distortion, num_parallel_calls=tf.data.AUTOTUNE)
    aug_ds = aug_ds.take(int(len(list(ds.as_numpy_iterator())) * fraction))

    # Combine original and augmented
    combined_ds = orig_ds.concatenate(aug_ds)
    return combined_ds.prefetch(tf.data.AUTOTUNE)


# =========================
# 3. Training loop
# =========================
def train_augmented(
    data_root: str,
    num_classes: int,
    batch_size: int = 64,
    epochs: int = 10,
    model_dir: str = "models",
    model_name: str = "classifier_augmented.keras",
    augment_fraction: float = 0.5,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:

    # --- Load raw waveform datasets ---
    train_wave_ds = load_waveform_dataset(data_root, split="train")
    val_wave_ds = load_waveform_dataset(data_root, split="val")

    # --- Combine originals with augmented samples ---
    train_wave_ds = augment_dataset(train_wave_ds, fraction=augment_fraction)

    # --- Turn waveforms into embeddings ---
    train_ds = make_embedding_dataset(train_wave_ds, batch_size=batch_size, shuffle=True)
    val_ds = make_embedding_dataset(val_wave_ds, batch_size=batch_size, shuffle=False)

    # --- Build classifier ---
    model = build_embedding_classifier(num_classes=num_classes)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=metrics)

    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, model_name),
            monitor="val_acc",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "logs"),
        ),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    final_path = os.path.join(model_dir, model_name.replace(".keras", "_final.keras"))
    model.save(final_path)
    print(f"Saved final augmented classifier to: {final_path}")

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default="classifier_augmented.keras")
    parser.add_argument("--augment_fraction", type=float, default=0.5,
                        help="Fraction of dataset to augment and append to originals")
    args = parser.parse_args()

    train_augmented(
        data_root=args.data_root,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=args.model_dir,
        model_name=args.model_name,
        augment_fraction=args.augment_fraction,
    )