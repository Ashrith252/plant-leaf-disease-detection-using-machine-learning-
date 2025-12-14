import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATA_DIR = os.path.join("data", "plant_multi")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 10

SEED = 42

os.makedirs("models", exist_ok=True)
RGB_SAVE = os.path.join("models", "cnn_rgb_multi.h5")
ENH_SAVE = os.path.join("models", "cnn_enhanced_multi.h5")


# ---------------- DATA GENERATORS ----------------

def create_rgb_generators():
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
        seed=SEED
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen


def apply_clahe_to_batch(batch):
    batch_uint8 = (batch * 255).astype(np.uint8)
    enhanced_batch = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img in batch_uint8:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        enhanced_batch.append(enhanced)

    enhanced_batch = np.array(enhanced_batch, dtype=np.float32) / 255.0
    return enhanced_batch


class ClaheGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_gen):
        self.base_gen = base_gen

    def __len__(self):
        return len(self.base_gen)

    def __getitem__(self, idx):
        x, y = self.base_gen[idx]
        x_enh = apply_clahe_to_batch(x)
        return x_enh, y

    @property
    def class_indices(self):
        return self.base_gen.class_indices

    @property
    def num_classes(self):
        return self.base_gen.num_classes


# ---------------- MODEL BUILDERS ----------------

def build_cnn_expert(input_shape, num_classes, name="cnn_expert"):
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
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------- TRAINING ----------------

def main():
    train_gen_rgb, val_gen_rgb = create_rgb_generators()
    num_classes = train_gen_rgb.num_classes
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

    print("Detected classes:", train_gen_rgb.class_indices)
    print("Num classes:", num_classes)

    train_gen_clahe = ClaheGenerator(train_gen_rgb)
    val_gen_clahe = ClaheGenerator(val_gen_rgb)

    cnn_rgb = build_cnn_expert(input_shape, num_classes, name="cnn_rgb_multi")
    cnn_enh = build_cnn_expert(input_shape, num_classes, name="cnn_enhanced_multi")

    chk_rgb = ModelCheckpoint(RGB_SAVE, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    chk_enh = ModelCheckpoint(ENH_SAVE, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    print("\nTraining CNN (RGB) ...")
    cnn_rgb.fit(
        train_gen_rgb,
        validation_data=val_gen_rgb,
        epochs=EPOCHS,
        callbacks=[chk_rgb, rlrop],
        verbose=1
    )

    print("\nTraining CNN (CLAHE-enhanced) ...")
    cnn_enh.fit(
        train_gen_clahe,
        validation_data=val_gen_clahe,
        epochs=EPOCHS,
        callbacks=[chk_enh, rlrop],
        verbose=1
    )

    print("\nTraining complete. Models saved to:")
    print(" ", RGB_SAVE)
    print(" ", ENH_SAVE)


if __name__ == "__main__":
    main()
