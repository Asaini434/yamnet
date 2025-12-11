import os
import json
import matplotlib.pyplot as plt

# Folders containing results
OG_RESULTS_ROOT = "../results/DistortionOGModelEval"
AUG_RESULTS_ROOT = "../results/DistortionAugmentedModelEval"

DISTORTIONS = ["noise", "pitch_shift", "time_stretch"]

for distortion in DISTORTIONS:
    # --- OG model ---
    og_folders = [f for f in os.listdir(OG_RESULTS_ROOT)
                  if (f.startswith(distortion) or f == "none_eval") and f.endswith("_eval")]
    og_levels, og_accs = [], []

    for folder in og_folders:
        if folder == "none_eval":
            level = 0.0
        else:
            level = float(folder.replace(distortion, "").replace("_eval", ""))
        results_file = os.path.join(OG_RESULTS_ROOT, folder, "results.json")
        if not os.path.exists(results_file):
            continue
        with open(results_file, "r") as f:
            acc = json.load(f).get("acc")
            if acc is not None:
                og_levels.append(level)
                og_accs.append(acc)

    if not og_levels:
        print(f"No OG results found for {distortion}")
        continue
    og_levels, og_accs = zip(*sorted(zip(og_levels, og_accs)))

    # --- Augmented model ---
    aug_folders = [f for f in os.listdir(AUG_RESULTS_ROOT)
                   if (f.startswith(f"aug_{distortion}") or f == "aug_none_eval") and f.endswith("_eval")]
    aug_levels, aug_accs = [], []

    for folder in aug_folders:
        if folder == "aug_none_eval":
            level = 0.0
        else:
            level = float(folder.replace(f"aug_{distortion}", "").replace("_eval", ""))
        results_file = os.path.join(AUG_RESULTS_ROOT, folder, "results.json")
        if not os.path.exists(results_file):
            continue
        with open(results_file, "r") as f:
            acc = json.load(f).get("acc")
            if acc is not None:
                aug_levels.append(level)
                aug_accs.append(acc)

    if not aug_levels:
        print(f"No augmented results found for {distortion}")
        continue
    aug_levels, aug_accs = zip(*sorted(zip(aug_levels, aug_accs)))

    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.plot(og_levels, og_accs, marker='o', label="Original Model")
    plt.plot(aug_levels, aug_accs, marker='s', label="Augmented Model")
    plt.xlabel(f"{distortion.replace('_',' ').title()} level")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs {distortion.replace('_',' ').title()}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Force 0 at origin
    max_level = max(max(og_levels), max(aug_levels))
    plt.xlim(0, max_level)

    # Save
    save_path = os.path.join(OG_RESULTS_ROOT, f"accuracy_vs_{distortion}_overlay.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved overlay plot for {distortion} to {save_path}")
    plt.show()