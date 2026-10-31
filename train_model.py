import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 30

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=42

)


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False

)


# ============================================================
# 4. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

NUM_CLASSES = len(class_names)

print("\n" + "=" * 70)
print("CLASS INFORMATION")
print("=" * 70)

for index, name in enumerate(class_names):

    print(f"{index} -> {name}")

print("\nTotal classes:", NUM_CLASSES)


# ============================================================
# 5. COUNT TRAINING IMAGES
# ============================================================

class_counts = {}

total_images = 0

for index, class_name in enumerate(class_names):

    class_path = os.path.join(TRAIN_DIR, class_name)

    count = 0

    for filename in os.listdir(class_path):

        file_path = os.path.join(class_path, filename)

        if os.path.isfile(file_path):

            count += 1

    class_counts[index] = count

    total_images += count


# ============================================================
# 6. BALANCED CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("BALANCED CLASS WEIGHTS")
print("=" * 70)

class_weights = {}

for index in range(NUM_CLASSES):

    count = class_counts[index]

    weight = total_images / (NUM_CLASSES * count)

    class_weights[index] = weight

    print(
        f"{index} -> "
        f"{class_names[index]:25s} : "
        f"{count:4d} images | "
        f"weight = {weight:.3f}"
    )


# ============================================================
# 7. PREFETCH DATA
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

data_augmentation = models.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.10
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# ============================================================
# 9. BUILD CUSTOM CNN
# ============================================================

model = models.Sequential([

    # Input
    layers.Input(
        shape=(224, 224, 3)
    ),

    # Data augmentation
    data_augmentation,

    # Normalize pixels
    layers.Rescaling(
        1.0 / 255
    ),


    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.Activation(
        "relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.Activation(
        "relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.Activation(
        "relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # BLOCK 4
    # --------------------------------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.Activation(
        "relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.40
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

], name="Arecanut_Custom_CNN")


# ============================================================
# 10. DISPLAY MODEL
# ============================================================

print("\n" + "=" * 70)
print("CUSTOM CNN ARCHITECTURE")
print("=" * 70)

model.summary()


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# 12. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    filepath=os.path.join(
        MODEL_DIR,
        "arecanut_custom_cnn_best.keras"
    ),

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=7,

    mode="max",

    restore_best_weights=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.00001,

    verbose=1

)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(

    train_dataset,

    validation_data=test_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]

)


# ============================================================
# 14. EVALUATE MODEL
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING MODEL ON TEST DATA")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print("\nTest Loss     :", test_loss)

print("Test Accuracy :", test_accuracy)


# ============================================================
# 15. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(

    MODEL_DIR,

    "arecanut_custom_cnn_final.keras"

)

model.save(
    final_model_path
)


# ============================================================
# 16. SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.txt"
)

with open(
    class_names_path,
    "w"
) as file:

    for name in class_names:

        file.write(
            name + "\n"
        )


# ============================================================
# 17. TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nFinal model:")
print(final_model_path)

print("\nClass names:")
print(class_names_path)

print("\nTest accuracy:")
print(test_accuracy)

print("\n" + "=" * 70)