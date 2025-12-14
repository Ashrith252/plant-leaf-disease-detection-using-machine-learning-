import os
import shutil

# existing roots (already split)
TOMATO_ROOT = os.path.join("plantvillage_tomato", "plantvillage")
POTATO_ROOT = "PlantVillage_potato"

# new combined root
COMBINED_ROOT = os.path.join("data", "plant_multi")
TRAIN_COMBINED = os.path.join(COMBINED_ROOT, "train")
VAL_COMBINED   = os.path.join(COMBINED_ROOT, "val")

os.makedirs(TRAIN_COMBINED, exist_ok=True)
os.makedirs(VAL_COMBINED, exist_ok=True)


def copy_classes(src_root, dst_root):
    """Copy train/ and val/ class folders from src_root to dst_root."""
    for split in ["train", "val"]:
        src_split = os.path.join(src_root, split)
        dst_split = os.path.join(dst_root, split)
        os.makedirs(dst_split, exist_ok=True)

        if not os.path.isdir(src_split):
            print(f"[WARNING] {src_split} does not exist, skipping.")
            continue

        class_folders = [
            d for d in os.listdir(src_split)
            if os.path.isdir(os.path.join(src_split, d))
        ]

        for cls in class_folders:
            src_cls_dir = os.path.join(src_split, cls)
            dst_cls_dir = os.path.join(dst_split, cls)
            os.makedirs(dst_cls_dir, exist_ok=True)

            for fname in os.listdir(src_cls_dir):
                src_fpath = os.path.join(src_cls_dir, fname)
                if not os.path.isfile(src_fpath):
                    continue
                dst_fpath = os.path.join(dst_cls_dir, fname)
                if not os.path.exists(dst_fpath):
                    shutil.copy(src_fpath, dst_fpath)

            print(f"Copied {split}/{cls} from {src_root} to {dst_root}")


def main():
    # copy tomato
    copy_classes(TOMATO_ROOT, COMBINED_ROOT)
    # copy potato
    copy_classes(POTATO_ROOT, COMBINED_ROOT)
    print("\nCombined dataset created at:", COMBINED_ROOT)


if __name__ == "__main__":
    main()
