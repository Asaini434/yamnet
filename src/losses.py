# src/train.py


# Right now this file is a working template with a placeholder dataset loader.
import os
import ssl
import certifi

# 1. Set environment variables BEFORE any imports
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# 2. Create custom SSL context using certifi
def create_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = create_ssl_context

# 3. Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~/.cache/tensorflow-hub')



import os
from typing import Tuple, List, Optional

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
tf.get_logger().setLevel('ERROR')

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

# focal loss function
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Simpler implementation that avoids the scoping issue
    """
    def loss(y_true, y_pred):
        # Softmax probabilities
        softmax_pred = tf.nn.softmax(y_pred)
        
        # Standard cross entropy
        ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        
        # Probability of true class
        p_t = tf.reduce_sum(y_true * softmax_pred, axis=-1)
        
        # Focal weight
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Alpha weighting (if provided)
        if alpha is not None:
            # Create alpha tensor
            if isinstance(alpha, (list, tuple)):
                alpha_t = tf.convert_to_tensor(alpha, dtype=tf.float32)
            else:
                alpha_t = tf.constant(alpha, dtype=tf.float32)
            
            # Apply alpha weighting per class
            alpha_factor = tf.reduce_sum(alpha_t * y_true, axis=-1)
            focal_weight = alpha_factor * focal_weight
        
        return tf.reduce_mean(focal_weight * ce)
    return loss


# =========================
# 2. Training loop
# =========================

def train(
    loss_fn: None,
    data_root: str,
    num_classes: int,
    batch_size: int = 64,
    epochs: int = 10,
    model_dir: str = "models",
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:

    # --- Load raw waveform datasets ---
    train_wave_ds = load_waveform_dataset(data_root, split="train")
    val_wave_ds = load_waveform_dataset(data_root, split="val")

    if loss_fn is not None:
        # --- This is for Categorical loss entropy (not Sparse)
        def to_one_hot(waveform, label):
            return waveform, tf.one_hot(label, depth=num_classes)

        train_wave_ds = train_wave_ds.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        val_wave_ds = val_wave_ds.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    # --- Turn waveforms into embeddings via YAMNet ---
    train_ds = make_embedding_dataset(train_wave_ds, batch_size=batch_size, shuffle=True)
    val_ds = make_embedding_dataset(val_wave_ds, batch_size=batch_size, shuffle=False)

    # --- Build classifier on top of embeddings ---
    model = build_embedding_classifier(num_classes=num_classes)

    # Standard loss for now;
    if loss_fn == None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ]
    else: 
        # --- This is for Categorical loss entropy and focal loss (not Sparse)
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),  # NOT SparseCategoricalAccuracy
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),  # NOT SparseTopK
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=metrics,
    )

    #os.makedirs(model_dir, exist_ok=True)

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

    # # Save final classifier
    # final_path = os.path.join(model_dir, "classifier_final_example.keras")
    # model.save(final_path)
    # print(f"Saved final classifier to: {final_path}")

    return model, history


if __name__ == "__main__":
    DATA_ROOT = "./../datasets/gtzan"
    NUM_CLASSES = 10
    EPOCHS = 5
    
    # Define loss functions
    loss_functions = {
        "Sparse CE": None,
        "CE + LS (α=0.1)": tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1 
        ),
        "Focal (γ=1.0, α=0.25)": focal_loss(gamma=1.0, alpha=0.25),
        "Focal (γ=2.0, α=0.25)": focal_loss(gamma=2.0, alpha=0.25)
    }
    
    # Store results
    results = {}  # Changed from 'data' to 'results' to avoid confusion
    
    # Train each loss function
    for name, loss_fn in loss_functions.items():
        print(f"\n{'='*50}")
        print(f"Training with: {name}")
        print(f"{'='*50}")
        
        try:
            model, history = train(
                loss_fn=loss_fn,
                data_root=DATA_ROOT,
                num_classes=NUM_CLASSES,
                batch_size=32,  # Added batch_size
                epochs=EPOCHS,
                model_dir="models"
            )
            
            # Store history
            results[name] = {
                'val_acc': history.history['val_acc'],
                'val_loss': history.history['val_loss'],
                'acc': history.history['acc'],
                'loss': history.history['loss']
            }
            
            print(f"✓ {name}: Final val_acc = {history.history['val_acc'][-1]:.4f}")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    with open('results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for name, data in results.items():
            json_results[name] = {
                k: [float(v) for v in vals] for k, vals in data.items()
            }
        json.dump(json_results, f, indent=2)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs_range = range(1, EPOCHS + 1)
    
    # Plot validation accuracy
    for name, data in results.items():
        ax1.plot(epochs_range, data['val_acc'], marker='o', linewidth=2, label=name)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Loss Function Comparison: Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation loss
    for name, data in results.items():
        ax2.plot(epochs_range, data['val_loss'], marker='s', linewidth=2, label=name)
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Loss Function Comparison: Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()