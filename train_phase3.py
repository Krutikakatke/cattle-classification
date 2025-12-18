"""
train_phase4_mac.py
PHASE 4 – FINAL ACCURACY PUSH (ULTRA CAREFUL)
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------
TRAIN_DIR = "/Users/krutikakatke/Documents/classification model/final_dataset/train"
VAL_DIR   = "/Users/krutikakatke/Documents/classification model/final_dataset/val"
OUT_DIR   = "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5              # 🔥 VERY IMPORTANT: DO NOT INCREASE
LR = 1e-5               # 🔥 Ultra-low learning rate
AUTOTUNE = tf.data.AUTOTUNE
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# -------- Mixed Precision --------
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

# -------- Load Phase-3 Model --------
phase3_model_path = os.path.join(OUT_DIR, "efficientnetv2-b0_phase3_final.keras")
model = keras.models.load_model(phase3_model_path, safe_mode=False)

# -------- Unfreeze TOP 95% of base model --------
base_model = model.layers[2]   # EfficientNet backbone
cut = int(len(base_model.layers) * 0.05)

for layer in base_model.layers[cut:]:
    layer.trainable = True

print(f"🔥 Unfrozen layers: {cut} → {len(base_model.layers)}")

# -------- Compile (LESS smoothing now) --------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

# -------- Callbacks --------
weights_path = os.path.join(OUT_DIR, "phase4_best_weights.weights.h5")

callbacks = [
    ModelCheckpoint(
        weights_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=1,
        min_lr=5e-7,
        verbose=1
    )
]

# -------- Train --------
print("\n🚀 PHASE 4 TRAINING STARTED\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -------- Save Final Model --------
final_model_path = os.path.join(OUT_DIR, "efficientnetv2-b0_phase4_final.keras")
model.save(final_model_path)

print("\n✅ PHASE 4 TRAINING FINISHED")
print("Best weights:", weights_path)
print("Final model:", final_model_path)