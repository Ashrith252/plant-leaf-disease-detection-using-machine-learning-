import os
import shutil
from sklearn.model_selection import train_test_split

# ====================
# CONFIG: PATHS
# ====================

# Tomato dataset root (we use the inner "plantvillage" folder)
TOMATO_DIR = os.path.join("plantvillage_tomato", "plantvillage")

# Potato dataset root
POTATO_DIR = "PlantVillage_potato"

# We will create train/ and val/ inside EACH of these
DATASET_ROOTS = [TOMATO_DIR, POTATO_DIR]

TEST_SIZE = 0.2   # 20% for validation
RANDOM_STATE = 42


def split_one_dataset(dataset_dir):
    print(f"\nProcessing dataset at: {dataset_dir}")

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List class folders (ignore train/ and val/ if they already exist)
    class_folders = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d not in ["train", "val"]
    ]

    print("Classes found:", class_folders)

    for cls in class_folders:
        class_path = os.path.join(dataset_dir, cls)
        images = [
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ]

        if len(images) == 0:
            print(f"  [WARNING] No images in {class_path}, skipping.")
            continue

        train_imgs, val_imgs = train_test_split(
            images, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Create class folders in train and val
        train_class_dir = os.path.join(train_dir, cls)
        val_class_dir = os.path.join(val_dir, cls)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy images
        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img),
                        os.path.join(train_class_dir, img))

        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img),
                        os.path.join(val_class_dir, img))

        print(f"  Class '{cls}': {len(train_imgs)} train, {len(val_imgs)} val")

    print(f"Finished splitting dataset at: {dataset_dir}")


def main():
    for root in DATASET_ROOTS:
        split_one_dataset(root)

    print("\nAll datasets processed successfully!")


if __name__ == "__main__":
    main()
