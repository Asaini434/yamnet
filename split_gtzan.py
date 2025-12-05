import pathlib
import random
import shutil

BASE = pathlib.Path("Data/genres_original") # Data folder contains original pre-split gtzan data
# jazz.00054 is broken in the original gtzan dataset; fixed version is attached
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
random.seed(69)

def main():
    genre_dirs = [
        d for d in BASE.iterdir()
        if d.is_dir() and d.name not in ("train", "val", "test")
    ]
    if not genre_dirs:
        raise RuntimeError("No genre directories found under datasets/gtzan")

    print("Genres detected:", [d.name for d in genre_dirs])

    for genre_dir in genre_dirs:
        genre = genre_dir.name
        print(f"\nProcessing genre: {genre}")
        files = sorted(genre_dir.glob("*.wav"))
        if not files:
            print(f"  WARNING: no .wav files found in {genre_dir}")
            continue

        random.shuffle(files)
        n = len(files)
        n_train = int(TRAIN_RATIO * n)
        n_val = int(VAL_RATIO * n)
        n_test = n - n_train - n_val
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        for split in ("train", "val", "test"):
            target_dir = BASE / split / genre
            target_dir.mkdir(parents=True, exist_ok=True)

        def copy_list(file_list, split):
            target_dir = BASE / split / genre
            for src in file_list:
                dst = target_dir / src.name
                shutil.copy2(src, dst)
        copy_list(train_files, "train")
        copy_list(val_files, "val")
        copy_list(test_files, "test")
        print(f"  {n_train} → train, {n_val} → val, {n_test} → test")

    print("\nDone, check train, val, test under datasets/gtzan/")

if __name__ == "__main__":
    main()