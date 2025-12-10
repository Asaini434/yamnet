import os
import subprocess

# Path to your eval script
EVAL_SCRIPT = "eval.py"

# Distortions to test
DISTORTIONS = ["noise", "gain", "time_stretch", "pitch_shift", "crop"]

# Common arguments
DATA_ROOT = "../datasets/gtzan"
MODEL_DIR = "../models/gtzan_baseline"
BATCH_SIZE = 64
SPLIT = "test"
TOPK = "3 5"
PLOT_CM = True

# Output root folder
OUTPUT_ROOT = "../results"

for distortion in DISTORTIONS:
    # Create a folder for each distortion
    output_dir = os.path.join(OUTPUT_ROOT, f"{distortion}_eval")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{distortion}.json")

    # Build the command
    cmd = [
        "python", EVAL_SCRIPT,
        "--data_root", DATA_ROOT,
        "--split", SPLIT,
        "--model_dir", MODEL_DIR,
        "--batch_size", str(BATCH_SIZE),
        "--output", output_file,
        "--distortion", distortion,
        "--topk", *TOPK.split()
    ]
    if PLOT_CM:
        cmd.append("--plot_cm")

    print(f"Running evaluation with distortion: {distortion}")
    subprocess.run(cmd)
    print(f"Finished {distortion}, results saved in {output_dir}\n")