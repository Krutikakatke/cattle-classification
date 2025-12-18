"""
train_phase1_mac_fixed.py

Phase 1: Transfer Learning (Feature Extraction)
- Load a pretrained base (EfficientNetV2-B0)
- Replace final head with a custom classifier for N_CLASSES (50)
- Freeze the base and train only the head
- Optimized for M-series Macs (mixed precision + data augmentation + TensorBoard)
- IMPORTANT: preprocessing is applied in the dataset pipeline (no Lambda layers)
"""

import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
)

# ----------------- USER CONFIG -----------------
TRAIN_DIR = "/Users/krutikakatke/Documents/classification model/final_dataset/train"
VAL_DIR   = "/Users/krutikakatke/Documents/classification model/final_dataset/val"
OUT_DIR   = "models"  # where models & logs will be saved
os.makedirs(OUT_DIR, exist_ok=True)

BASE_MODEL_NAME = "efficientnetv2-b0"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
N_CLASSES = 50
PHASE1_EPOCHS = 8
LEARNING_RATE = 1e-4
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
# ------------------------------------------------

# ----------------- Mixed Precision -----------------
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# ----------------- GPU memory growth (prevent crashes) -----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth")
    except RuntimeError as e:
        print(e)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ----------------- Utilities -----------------
def get_preprocess_and_model(name, input_shape):
    name = name.lower()
    if name == "efficientnetv2-b0":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, EfficientNetV2B0
        base = EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {name}")
    return preprocess_input, base

# ----------------- Prepare datasets (apply preprocess_input here to avoid Lambda) -----------------
IMG_H, IMG_W = IMAGE_SIZE
input_shape = (IMG_H, IMG_W, 3)

print("Preparing datasets from directories...")

# create raw datasets first (uint8 images)
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=False,
    seed=SEED
)

# get preprocess function & base model (so we can call preprocess_input)
preprocess_input, base_model = get_preprocess_and_model(BASE_MODEL_NAME, input_shape)

# Data augmentation (apply as part of dataset pipeline or model; here we keep it in model for convenience)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# Apply preprocessing (preprocess_input expects float images; map handles it)
def train_preprocess(images, labels):
    # images: uint8 -> preprocess_input will convert/normalize appropriately
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)   # apply EfficientNetV2 preprocessing (no Lambda layer in model)
    return images, labels

def val_preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels

train_ds = raw_train_ds.map(train_preprocess, num_parallel_calls=AUTOTUNE)
val_ds   = raw_val_ds.map(val_preprocess, num_parallel_calls=AUTOTUNE)

# Optionally, apply augmentation in the dataset pipeline as well (uncomment if desired)
# def augment(images, labels):
#     images = data_augmentation(images, training=True)
#     return images, labels
# train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

# ----------------- Sanity check -----------------
CLASS_NAMES = raw_train_ds.class_names
print("Detected classes ({}):".format(len(CLASS_NAMES)), CLASS_NAMES)
if len(CLASS_NAMES) != N_CLASSES:
    print(f"Warning: N_CLASSES={N_CLASSES} but found {len(CLASS_NAMES)} classes in TRAIN_DIR.")

# ----------------- Prefetch / cache -----------------
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----------------- Build model (no Lambda layer anywhere) -----------------
print(f"Building model with base: {BASE_MODEL_NAME}")

# Freeze base for Phase 1 (we already created base_model above)
base_model.trainable = False

# Model head (apply augmentation here so model sees random transforms)
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)                # augmentation inside model is fine
x = base_model(x, training=False)            # base expects preprocessed inputs; our dataset is preprocessed already
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(N_CLASSES, activation="softmax", dtype="float32")(x)

model = keras.Model(inputs, outputs, name=f"{BASE_MODEL_NAME}_phase1")
model.summary()

# ----------------- Compile -----------------
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------- Callbacks -----------------
checkpoint_path = os.path.join(OUT_DIR, f"{BASE_MODEL_NAME}_phase1_best.h5")
log_dir = os.path.join(OUT_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    CSVLogger(os.path.join(OUT_DIR, f"{BASE_MODEL_NAME}_phase1_log.csv")),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# ----------------- Train (Phase 1: Head Only) -----------------
print("Starting Phase 1 training (training only the head)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    callbacks=callbacks
)

# ----------------- Save final Phase 1 model -----------------
final_path = os.path.join(OUT_DIR, f"{BASE_MODEL_NAME}_phase1_final.h5")
model.save(final_path)
print(f"Phase 1 completed. Best checkpoint: {checkpoint_path}, final saved model: {final_path}")
