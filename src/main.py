
import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR  = os.path.join("data", "plant_multi")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")

# ======================
# CONFIG
# ======================

# For now we train on TOMATO dataset only
DATA_DIR = os.path.join("plantvillage_tomato", "plantvillage")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5          # later you can increase to 20–30
SEED = 42


# ======================
# 1. DATA GENERATORS
# ======================

def create_rgb_generators():
    """Standard RGB generators for CNN Expert 1."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen


def apply_clahe_to_batch(batch):
    """
    Apply CLAHE to a batch of RGB images for CNN Expert 2.
    Input batch is float32 [0,1] from ImageDataGenerator.
    """
    batch_uint8 = (batch * 255).astype(np.uint8)
    enhanced_batch = []

    # CLAHE settings
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img in batch_uint8:
        # img is RGB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        enhanced_batch.append(enhanced)

    enhanced_batch = np.array(enhanced_batch, dtype=np.float32) / 255.0
    return enhanced_batch


class ClaheGenerator(tf.keras.utils.Sequence):
    """
    Wraps an existing generator and applies CLAHE to images on the fly.
    This will be used for CNN Expert 2.
    """

    def __init__(self, base_gen):
        self.base_gen = base_gen

    def __len__(self):
        return len(self.base_gen)

    def __getitem__(self, idx):
        x, y = self.base_gen[idx]
        x_enh = apply_clahe_to_batch(x)
        return x_enh, y

    @property
    def classes(self):
        return self.base_gen.classes

    @property
    def class_indices(self):
        return self.base_gen.class_indices


# ======================
# 2. CNN MODELS
# ======================

def build_cnn_expert(input_shape, num_classes, name="cnn_expert"):
    """
    Generic CNN model; used for both experts.
    """
    inputs = layers.Input(shape=input_shape, name=name + "_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name=name + "_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ======================
# 3. MAIN PIPELINE
# ======================

def main():
    # 1. Create base RGB generators
    train_gen_rgb, val_gen_rgb = create_rgb_generators()

    num_classes = train_gen_rgb.num_classes
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

    print("Classes:", train_gen_rgb.class_indices)

    # 2. Wrap for CLAHE generator (for enhanced CNN expert)
    train_gen_clahe = ClaheGenerator(train_gen_rgb)
    val_gen_clahe = ClaheGenerator(val_gen_rgb)

    # 3. Build CNN Expert 1 (RGB)
    cnn_rgb = build_cnn_expert(input_shape, num_classes, name="cnn_rgb")
    print(cnn_rgb.summary())

    # 4. Build CNN Expert 2 (Enhanced / CLAHE)
    cnn_enh = build_cnn_expert(input_shape, num_classes, name="cnn_enhanced")
    print(cnn_enh.summary())

    # 5. Train CNN Expert 1 (RGB)
    print("\nTraining CNN Expert 1 (RGB) ...")
    cnn_rgb.fit(
        train_gen_rgb,
        validation_data=val_gen_rgb,
        epochs=EPOCHS,
    )

    # 6. Train CNN Expert 2 (CLAHE)
    print("\nTraining CNN Expert 2 (Enhanced / CLAHE) ...")
    cnn_enh.fit(
        train_gen_clahe,
        validation_data=val_gen_clahe,
        epochs=EPOCHS,
    )

    # 7. Save models
    os.makedirs("models", exist_ok=True)
    cnn_rgb.save(os.path.join("models", "cnn_rgb_tomato.h5"))
    cnn_enh.save(os.path.join("models", "cnn_enhanced_tomato.h5"))
    print("\nSaved CNN models in 'models/' folder.")


if __name__ == "__main__":
    main()
