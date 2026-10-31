import tensorflow as tf
import os

# ==========================================
# SETTINGS
# ==========================================

TRAIN_DIR = "dataset/train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# LOAD TRAIN DATASET
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================================
# DISPLAY CLASS INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("ARECANUT CUSTOM CNN - CLASS CHECK")
print("=" * 60)

print("\nClass names detected by TensorFlow:")

for index, class_name in enumerate(train_dataset.class_names):
    print(f"{index}  ->  {class_name}")

print("\n" + "=" * 60)

print(f"Total classes: {len(train_dataset.class_names)}")

print("=" * 60)

# ==========================================
# CHECK NUMBER OF IMAGES
# ==========================================

total_images = 0

for images, labels in train_dataset:
    total_images += images.shape[0]

print(f"\nTotal training images loaded: {total_images}")

print("=" * 60)

# ==========================================
# CHECK LABELS IN FIRST BATCH
# ==========================================

for images, labels in train_dataset.take(1):

    print("\nFirst batch label numbers:")

    print(labels.numpy())

    print("\nFirst batch image shape:")
    print(images.shape)

print("\n" + "=" * 60)
print("CLASS CHECK COMPLETED")
print("=" * 60)