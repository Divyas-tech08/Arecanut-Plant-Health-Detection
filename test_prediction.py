import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# ============================================================
# 1. MODEL PATH
# ============================================================

MODEL_PATH = "model/arecanut_custom_cnn_final.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# 2. LOAD CLASS NAMES
# ============================================================

with open("model/class_names.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# 3. IMAGE PATH
# ============================================================

IMAGE_PATH = input("\nEnter image path: ").strip()


# ============================================================
# 4. LOAD IMAGE
# ============================================================

img = image.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ============================================================
# 5. PREDICTION
# ============================================================

predictions = model.predict(img_array)

predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100


# ============================================================
# 6. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("Predicted Class :", predicted_class)
print("Confidence      :", f"{confidence:.2f}%")

print("=" * 60)