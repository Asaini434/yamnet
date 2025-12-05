import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score,
)

from train import load_waveform_dataset, make_embedding_dataset, CLASS_NAMES
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root of dataset, e.g. datasets/gtzan")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--model_dir", type=str, default="models", help="Directory containing classifier_best.keras")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", type=str, default="results/eval_gtzan.json")
    ap.add_argument("--distortion", type=str, default="none", choices=["none", "noise", "gain"])
    ap.add_argument("--topk", type=int, nargs="*", default=[3, 5])
    ap.add_argument("--plot_cm", action="store_true", help="Plot confusion matrix")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    base_ds = load_waveform_dataset(args.data_root, split=args.split, shuffle=False)

    def _apply_distortion(waveform, label):
        if args.distortion == "none":
            return waveform, label
        elif args.distortion == "noise":
            rms = tf.sqrt(tf.reduce_mean(tf.square(waveform)) + 1e-8)
            noise = tf.random.normal(tf.shape(waveform), stddev=rms / 10.0)
            return tf.clip_by_value(waveform + noise, -1.0, 1.0), label
        elif args.distortion == "gain":
            gain_db = tf.random.uniform([], -6.0, 6.0)
            gain = tf.pow(10.0, gain_db / 20.0)
            return tf.clip_by_value(waveform * gain, -1.0, 1.0), label
        return waveform, label

    wave_ds = base_ds.map(_apply_distortion, num_parallel_calls=tf.data.AUTOTUNE)
    wave_ds = wave_ds.prefetch(tf.data.AUTOTUNE)
    emb_ds = make_embedding_dataset(wave_ds, batch_size=args.batch_size, shuffle=False)
    model_path = os.path.join(args.model_dir, "classifier_best.keras") # load model inline
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path)

    all_logits = [] # collect logits and labels inline
    all_labels = []
    for batch_emb, batch_labels in emb_ds:
        logits = model(batch_emb, training=False)
        all_logits.append(logits.numpy())
        all_labels.append(batch_labels.numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    num_classes = all_logits.shape[1]

    preds = all_logits.argmax(axis=1) # compute metrics inline
    acc = accuracy_score(all_labels, preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="macro", zero_division=0
    )
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(
        all_labels, preds, average=None, zero_division=0
    )
    cm = confusion_matrix(all_labels, preds, labels=np.arange(num_classes))

    metrics = {
        "acc": float(acc),
        "macro_precision": float(macro_prec),"macro_recall": float(macro_rec),"macro_f1": float(macro_f1),
        "per_class_precision": per_prec.tolist(),"per_class_recall": per_rec.tolist(),"per_class_f1": per_f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    for k in args.topk:
        if k <= num_classes:
            metrics[f"top{k}_acc"] = float(
                top_k_accuracy_score(all_labels, all_logits, k=k)
            )
    if CLASS_NAMES is not None:
        metrics["class_names"] = CLASS_NAMES
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Split:", args.split)
    print("Distortion:", args.distortion)
    print("Accuracy:", metrics["acc"])
    print("Macro F1:", metrics["macro_f1"])

    if args.plot_cm and CLASS_NAMES is not None: # plot confusion matrix if specified in params
        cm_plot_path = args.output.replace(".json", "_confusion_matrix.png")
        plt.figure(figsize=(12, 10))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_normalized,
            annot=True,fmt='.2f',cmap='Blues',
            xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,cbar_kws={'label': 'Proportion'}
        )
        plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {cm_plot_path}")
        plt.close()