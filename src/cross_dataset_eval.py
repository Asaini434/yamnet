import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from train import load_waveform_dataset, make_embedding_dataset, CLASS_NAMES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_root", type=str, default="datasets/gtzan")
    ap.add_argument("--target_root", type=str, required=True)
    ap.add_argument("--source_split", type=str, default="test")
    ap.add_argument("--target_split", type=str, default="test")
    ap.add_argument("--model_dir", type=str, default="models/gtzan_baseline")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output_dir", type=str, default="results/cross_dataset")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--max_iter", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "classifier_best.keras")
    model = tf.keras.models.load_model(model_path)

    wave_ds = load_waveform_dataset(args.source_root, split=args.source_split, shuffle=False) # evaluate source dataset
    emb_ds = make_embedding_dataset(wave_ds, batch_size=args.batch_size, shuffle=False)
    src_embeddings = []
    src_logits = []
    src_labels = []
    for batch_emb, batch_labels in emb_ds:
        logits = model(batch_emb, training=False)
        src_embeddings.append(batch_emb.numpy())
        src_logits.append(logits.numpy())
        src_labels.append(batch_labels.numpy())
    src_embeddings = np.concatenate(src_embeddings, axis=0)
    src_logits = np.concatenate(src_logits, axis=0)
    src_labels = np.concatenate(src_labels, axis=0)
    src_preds = src_logits.argmax(axis=1)
    src_acc = accuracy_score(src_labels, src_preds)
    _, _, src_f1, _ = precision_recall_fscore_support(src_labels, src_preds, average='macro', zero_division=0)
    src_sil = silhouette_score(src_embeddings, src_labels)
    src_db = davies_bouldin_score(src_embeddings, src_labels)
    src_ch = calinski_harabasz_score(src_embeddings, src_labels)
    source_metrics = {
        'accuracy': float(src_acc),'macro_f1': float(src_f1),
        'silhouette_score': float(src_sil),'davies_bouldin_index': float(src_db),
        'calinski_harabasz_score': float(src_ch),'num_samples': int(len(src_labels))
    }
    print(f"Source: acc={src_acc:.4f}, f1={src_f1:.4f}, sil={src_sil:.4f}")

    wave_ds = load_waveform_dataset(args.target_root, split=args.target_split, shuffle=False) # evaluate target dataset
    emb_ds = make_embedding_dataset(wave_ds, batch_size=args.batch_size, shuffle=False)
    tgt_embeddings = []
    tgt_logits = []
    tgt_labels = []
    for batch_emb, batch_labels in emb_ds:
        logits = model(batch_emb, training=False)
        tgt_embeddings.append(batch_emb.numpy())
        tgt_logits.append(logits.numpy())
        tgt_labels.append(batch_labels.numpy())
    tgt_embeddings = np.concatenate(tgt_embeddings, axis=0)
    tgt_logits = np.concatenate(tgt_logits, axis=0)
    tgt_labels = np.concatenate(tgt_labels, axis=0)
    tgt_preds = tgt_logits.argmax(axis=1)
    tgt_acc = accuracy_score(tgt_labels, tgt_preds)
    _, _, tgt_f1, _ = precision_recall_fscore_support(tgt_labels, tgt_preds, average='macro', zero_division=0)
    tgt_sil = silhouette_score(tgt_embeddings, tgt_labels)
    tgt_db = davies_bouldin_score(tgt_embeddings, tgt_labels)
    tgt_ch = calinski_harabasz_score(tgt_embeddings, tgt_labels)
    import pathlib
    target_split_dir = pathlib.Path(args.target_root) / args.target_split
    target_class_names = sorted([d.name for d in target_split_dir.iterdir() if d.is_dir()])
    print(f"Target: acc={tgt_acc:.4f}, f1={tgt_f1:.4f}, sil={tgt_sil:.4f}")
    print(f"Drop: acc={src_acc - tgt_acc:+.4f}, f1={src_f1 - tgt_f1:+.4f}")
    target_metrics = {
        'accuracy': float(tgt_acc),'macro_f1': float(tgt_f1),
        'silhouette_score': float(tgt_sil),'davies_bouldin_index': float(tgt_db),
        'calinski_harabasz_score': float(tgt_ch),'num_samples': int(len(tgt_labels))
    }
    results = {
        'source_dataset': args.source_root,'target_dataset': args.target_root,
        'source_metrics': source_metrics,'target_metrics': target_metrics,
        'performance_drop': {
            'accuracy': float(src_acc - tgt_acc),
            'macro_f1': float(src_f1 - tgt_f1),
            'silhouette_score': float(src_sil - tgt_sil)
        }
    }
    results_path = os.path.join(args.output_dir, "cross_dataset_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    combined_embeddings = np.vstack([src_embeddings, tgt_embeddings]) # combined t-SNE visualization
    combined_labels = np.concatenate([src_labels, tgt_labels])
    dataset_source = np.array(['GTZAN'] * len(src_labels) + ['FMA'] * len(tgt_labels))
    tsne = TSNE(
        n_components=2,perplexity=args.perplexity,
        max_iter=args.max_iter,init="random",
        learning_rate="auto",random_state=42
    )
    coords = tsne.fit_transform(combined_embeddings)
    src_coords = coords[:len(src_labels)]
    tgt_coords = coords[len(src_labels):]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    colors_src = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = src_labels == class_idx
        axes[0].scatter(
            src_coords[mask, 0], src_coords[mask, 1],
            c=[colors_src[class_idx]], label=class_name,
            alpha=0.6, s=50, edgecolors='black', linewidth=0.5
        )
    axes[0].set_title(f'Source: GTZAN (n={len(src_labels)})', fontsize=14)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3)
    colors_tgt = plt.cm.tab10(np.linspace(0, 1, len(target_class_names)))
    for class_idx, class_name in enumerate(target_class_names):
        mask = tgt_labels == class_idx
        axes[1].scatter(
            tgt_coords[mask, 0], tgt_coords[mask, 1],
            c=[colors_tgt[class_idx]], label=class_name,
            alpha=0.6, s=50, edgecolors='black', linewidth=0.5
        )
    axes[1].set_title(f'Target: FMA (n={len(tgt_labels)})', fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "cross_dataset_tsne.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    plt.figure(figsize=(12, 10))

    src_mask = dataset_source == 'GTZAN'
    tgt_mask = dataset_source == 'FMA'
    plt.scatter(coords[src_mask, 0], coords[src_mask, 1],
                c='steelblue', label='GTZAN', alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    plt.scatter(coords[tgt_mask, 0], coords[tgt_mask, 1],
                c='darkorange', label='FMA', alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    plt.title('Cross-Dataset Embedding Space (Combined t-SNE)', fontsize=16, pad=20)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    overlay_path = os.path.join(args.output_dir, "cross_dataset_overlay.png")
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {overlay_path}")
    plt.close()
