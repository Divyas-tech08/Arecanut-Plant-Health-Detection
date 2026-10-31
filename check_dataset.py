import os
from PIL import Image

DATASET_DIR = "dataset"

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("=" * 60)
print("ARECANUT DATASET CHECK")
print("=" * 60)

total_images = 0
total_errors = 0

for split in ["train", "test"]:

    split_path = os.path.join(DATASET_DIR, split)

    print(f"\n--- {split.upper()} DATASET ---")

    for class_name in sorted(os.listdir(split_path)):

        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        class_count = 0
        class_errors = 0

        for filename in os.listdir(class_path):

            file_path = os.path.join(class_path, filename)

            if not filename.lower().endswith(valid_extensions):
                continue

            try:
                with Image.open(file_path) as image:
                    image.verify()

                class_count += 1

            except Exception:
                class_errors += 1
                total_errors += 1

        total_images += class_count

        print(
            f"{class_name:25} "
            f"{class_count:5} valid images | "
            f"{class_errors:3} errors"
        )

print("\n" + "=" * 60)
print(f"TOTAL VALID IMAGES : {total_images}")
print(f"TOTAL IMAGE ERRORS : {total_errors}")
print("=" * 60)