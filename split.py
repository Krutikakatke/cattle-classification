import os
import shutil
from sklearn.model_selection import train_test_split

# ---- PATHS ----
SOURCE_DATASET = "/Users/krutikakatke/Documents/classification model/cattle"         # Your folder with 50 subfolders (each = breed)
DEST_DATASET = "final_dataset"    # Output folder

# ---- SPLIT RATIOS ----
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1   # Remaining

# ---- CREATE DESTINATION FOLDERS ----
for split in ["train", "val", "test"]:
    split_path = os.path.join(DEST_DATASET, split)
    os.makedirs(split_path, exist_ok=True)


# ---- PROCESS EACH BREED ----
breed_names = os.listdir(SOURCE_DATASET)

for breed in breed_names:
    breed_path = os.path.join(SOURCE_DATASET, breed)

    if not os.path.isdir(breed_path):
        continue

    print(f"Processing breed: {breed}")

    # Create breed subfolders inside train/val/test
    os.makedirs(os.path.join(DEST_DATASET, "train", breed), exist_ok=True)
    os.makedirs(os.path.join(DEST_DATASET, "val", breed), exist_ok=True)
    os.makedirs(os.path.join(DEST_DATASET, "test", breed), exist_ok=True)

    # Collect all image file paths
    images = [
        f for f in os.listdir(breed_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # ---- TRAIN / TEMP SPLIT ----
    train_imgs, temp_imgs = train_test_split(
        images, train_size=TRAIN_SPLIT, shuffle=True, random_state=42
    )

    # ---- VAL / TEST SPLIT ----
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)

    val_imgs, test_imgs = train_test_split(
        temp_imgs, train_size=val_ratio, shuffle=True, random_state=42
    )

    # ---- COPY FILES ----
    for img in train_imgs:
        shutil.copy(
            os.path.join(breed_path, img),
            os.path.join(DEST_DATASET, "train", breed, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(breed_path, img),
            os.path.join(DEST_DATASET, "val", breed, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(breed_path, img),
            os.path.join(DEST_DATASET, "test", breed, img)
        )

print("\nDataset split completed successfully!")
print("Check the final_dataset/ folder.")
