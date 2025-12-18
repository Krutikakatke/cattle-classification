"""
train_phase2_mac.py
STABLE VERSION – NO CRASH
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------
TRAIN_DIR = "/Users/krutikakatke/Documents/classification model/final_dataset/train"
VAL_DIR   = "/Users/krutikakatke/Documents/classification model/final_dataset/val"
OUT_DIR   = "models"

BASE_MODEL_NAME = "efficientnetv2-b0"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 6
LR = 1e-5
AUTOTUNE = tf.data.AUTOTUNE
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# -------- Mixed Precision (SAFE) --------
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# -------- GPU config --------
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# -------- Dataset --------
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

def preprocess(x, y):
    return preprocess_input(tf.cast(x, tf.float32)), y

train_ds = train_ds.map(preprocess).prefetch(AUTOTUNE)
val_ds   = val_ds.map(preprocess).prefetch(AUTOTUNE)

# -------- Load Phase-1 Model --------
phase1_path = os.path.join(OUT_DIR, f"{BASE_MODEL_NAME}_phase1_best.h5")
model = keras.models.load_model(phase1_path, safe_mode=False)

# -------- Unfreeze top 40% --------
base_model = model.layers[2]
cut = int(len(base_model.layers) * 0.6)

for layer in base_model.layers[cut:]:
    layer.trainable = True

# -------- Compile --------
model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------- Callbacks (SAFE ONLY) --------
weights_path = os.path.join(OUT_DIR, "phase2_best_weights.weights.h5")

callbacks = [
    ModelCheckpoint(
        weights_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,   # 🔑 THIS FIXES EVERYTHING
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
]

# -------- Train --------
print("\n🚀 PHASE 2 TRAINING STARTED\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -------- Save FINAL MODEL MANUALLY --------
final_model_path = os.path.join(OUT_DIR, "efficientnetv2-b0_phase2_final.keras")

model.save(final_model_path)

print("\n✅ TRAINING FINISHED")
print("Best weights:", weights_path)
print("Final model:", final_model_path)
