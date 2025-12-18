import os

# Path to your 50 breed folders
BASE_DIR = "/Users/krutikakatke/Documents/classification model/cattle"

# Allowed image extensions
ALLOWED_EXT = ["jpg", "jpeg", "png", "webp", "jfif"]

# Loop through each breed folder
for breed in os.listdir(BASE_DIR):
    breed_path = os.path.join(BASE_DIR, breed)

    if not os.path.isdir(breed_path):
        continue

    print(f"Renaming images in: {breed}")

    images = [
        f for f in os.listdir(breed_path)
        if f.split(".")[-1].lower() in ALLOWED_EXT
    ]

    # Sort for consistent naming order
    images.sort()

    # Rename files with incremental numbering
    for idx, img in enumerate(images, start=1):
        ext = img.split(".")[-1].lower()
        new_name = f"{breed}_{idx:04d}.{ext}"     # example → deoni_0001.jpg
        
        old_path = os.path.join(breed_path, img)
        new_path = os.path.join(breed_path, new_name)

        os.rename(old_path, new_path)

print("\nRenaming completed for all breeds!")
