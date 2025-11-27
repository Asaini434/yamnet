# src/yamnet_backbone.py

import tensorflow as tf
import tensorflow_hub as hub

# Official TF Hub YAMNet handle
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

# Global cached model 
_yamnet_model = None


def get_yamnet_model():
    global _yamnet_model
    if _yamnet_model is None:
        _yamnet_model = hub.load(YAMNET_HANDLE)
    return _yamnet_model


@tf.function
def yamnet_infer(waveform: tf.Tensor):
    """
    Run YAMNet on a single waveform.

    Args:
        waveform: 1D float32 tensor, 16 kHz mono, values in [-1, 1].

    Returns:
        scores:      [num_frames, 521]
        embeddings:  [num_frames, 1024]
        spectrogram: [num_frames, num_mel_bins]
    """
    model = get_yamnet_model()
    waveform = tf.reshape(waveform, [-1])  # make sure it's 1-D
    scores, embeddings, spectrogram = model(waveform)
    return scores, embeddings, spectrogram


@tf.function
def waveform_to_embedding(waveform: tf.Tensor) -> tf.Tensor:
    _, embeddings, _ = yamnet_infer(waveform)
    # Average over time frames
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding


def map_to_embedding(example_waveform, example_label):
    """
    Helper for tf.data: (waveform, label) -> (embedding, label)
    """
    embedding = waveform_to_embedding(example_waveform)
    return embedding, example_label
