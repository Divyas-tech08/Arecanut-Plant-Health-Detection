import os

# ==========================================
# SETTINGS
# ==========================================

TRAIN_DIR = "dataset/train"


# ==========================================
# GET CLASS NAMES
# ==========================================

class_names = sorted(
    [
        folder
        for folder in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, folder))
    ]
)


# ==========================================
# COUNT IMAGES IN EACH CLASS
# ==========================================

class_counts = {}

for class_name in class_names:

    class_path = os.path.join(TRAIN_DIR, class_name)

    count = 0

    for filename in os.listdir(class_path):

        file_path = os.path.join(class_path, filename)

        if os.path.isfile(file_path):
            count += 1

    class_counts[class_name] = count


# ==========================================
# FIND LARGEST CLASS
# ==========================================

max_count = max(class_counts.values())


# ==========================================
# CALCULATE CLASS WEIGHTS
# ==========================================

class_weights = {}

for index, class_name in enumerate(class_names):

    count = class_counts[class_name]

    weight = max_count / count

    class_weights[index] = weight


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 70)
print("ARECANUT CUSTOM CNN - CLASS BALANCE CHECK")
print("=" * 70)

print("\nClass distribution:\n")

for index, class_name in enumerate(class_names):

    print(
        f"{index:2d} -> "
        f"{class_name:25s} : "
        f"{class_counts[class_name]:4d} images "
        f"| weight = {class_weights[index]:.2f}"
    )


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 70)

print(f"Total images   : {sum(class_counts.values())}")
print(f"Total classes  : {len(class_names)}")
print(f"Largest class  : {max_count} images")

print("=" * 70)