import os
import argparse
import csv
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from train import load_waveform_dataset, make_embedding_dataset, CLASS_NAMES

if __name__ == "__main__":
    ap = argparse.ArgumentParser() # parse arguments inline
    ap.add_argument("--data_root", type=str, required=True, help="Root of dataset, e.g. datasets/gtzan")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output_csv", type=str, default="results/tsne_gtzan_test.csv")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n_iter", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    wave_ds = load_waveform_dataset(args.data_root, split=args.split, shuffle=False)
    emb_ds = make_embedding_dataset(wave_ds, batch_size=args.batch_size, shuffle=False)
    all_emb = [] # collect embeddings and labels inline
    all_labels = []
    for batch_emb, batch_labels in emb_ds:
        all_emb.append(batch_emb.numpy())
        all_labels.append(batch_labels.numpy())
    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print("Embeddings shape:", embeddings.shape)

    sil_score = silhouette_score(embeddings, labels) # compute cluster quality metrics
    db_index = davies_bouldin_score(embeddings, labels)
    ch_score = calinski_harabasz_score(embeddings, labels)
    print(f"Silhouette: {sil_score:.4f}, Davies-Bouldin: {db_index:.4f}, Calinski-Harabasz: {ch_score:.2f}")
    cluster_metrics = {
        'silhouette_score': float(sil_score),
        'davies_bouldin_index': float(db_index),
        'calinski_harabasz_score': float(ch_score)
    }
    metrics_path = args.output_csv.replace(".csv", "_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(cluster_metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    tsne = TSNE(n_components=2,perplexity=args.perplexity,max_iter=args.n_iter,init="random",learning_rate="auto",)
    coords = tsne.fit_transform(embeddings)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "class_name"])
        for (x, y), label in zip(coords, labels):
            cname = CLASS_NAMES[label] if CLASS_NAMES is not None else str(label)
            writer.writerow([float(x), float(y), int(label), cname])
    print(f"Saved t-SNE coordinates to {args.output_csv}")

    if CLASS_NAMES is not None: # t-SNE plot
        plot_path = args.output_csv.replace(".csv", ".png")
        plt.figure(figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
        for class_idx, class_name in enumerate(CLASS_NAMES):
            mask = labels == class_idx
            plt.scatter(coords[mask, 0], coords[mask, 1], c=[colors[class_idx]],
                        label=class_name, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        plt.title('t-SNE Visualization', fontsize=16, pad=20)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        plt.close()