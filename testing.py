# =========================
# TEST SET CONFUSION MATRIX
# =========================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


import tensorflow as tf

# 🔴 CHANGE THIS to your actual model file
MODEL_PATH = "/Users/krutikakatke/Documents/classification model/website/efficientnetv2-b0_phase3_final.keras"

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")


# 🔴 CHANGE THIS PATH
TEST_DIR = "/Users/krutikakatke/Documents/classification model/final_dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1️⃣ Load TEST dataset (never used in training)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

CLASS_NAMES = test_ds.class_names
print("Classes:", CLASS_NAMES)

# 2️⃣ Predict on test data
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# 3️⃣ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 12))
plt.imshow(cm)
plt.title("Confusion Matrix (Test Set)")
plt.colorbar()

ticks = np.arange(len(CLASS_NAMES))
plt.xticks(ticks, CLASS_NAMES, rotation=90)
plt.yticks(ticks, CLASS_NAMES)

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()