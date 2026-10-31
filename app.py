import os
import csv
import numpy as np
import tensorflow as tf

from history import (
    init_database,
    save_prediction,
    get_history,
    get_condition_history,
    clear_history
)
from flask import Flask, render_template, request, send_file,redirect, url_for
from tensorflow.keras.utils import load_img, img_to_array

from disease_info import DISEASE_INFO
from translations import TRANSLATIONS


# ============================================================
# 1. FLASK APPLICATION
# ============================================================

app = Flask(__name__)
# Initialize prediction history database
init_database()


# ============================================================
# 2. PATH SETTINGS
# ============================================================

MODEL_PATH = "model/arecanut_custom_cnn_best.keras"
CLASS_NAMES_PATH = "model/class_names.txt"

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# 3. PREDICTION CONFIDENCE THRESHOLD
# ============================================================

CONFIDENCE_THRESHOLD = 50.0


# ============================================================
# 4. LOAD TRAINED CNN MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING ARECANUT CUSTOM CNN MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 5. LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as file:

    class_names = [
        line.strip()
        for line in file.readlines()
    ]


print("\nClasses:")

for index, name in enumerate(class_names):

    print(f"{index} -> {name}")


print("\nTotal classes:", len(class_names))


# ============================================================
# 6. IMAGE PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_img(
        image_path,
        target_size=(224, 224)
    )


    # --------------------------------------------------------
    # Convert image to array
    # --------------------------------------------------------

    image_array = img_to_array(image)


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )


    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_index = np.argmax(
        predictions[0]
    )


    # --------------------------------------------------------
    # Get confidence internally
    # --------------------------------------------------------

    confidence = float(
        predictions[0][predicted_index]
    ) * 100


    predicted_class = class_names[predicted_index]


    # --------------------------------------------------------
    # CONFIDENCE CHECK
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        return None


    return predicted_class


# ============================================================
# 7. HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    image_path = None

    disease_info = None

    is_healthy = False

    is_unknown = False

    error = None

    language = "English"


    # ========================================================
    # POST REQUEST
    # ========================================================

    if request.method == "POST":


        # ----------------------------------------------------
        # Get selected language
        # ----------------------------------------------------

        language = request.form.get(
            "language",
            "English"
        )


        if language not in TRANSLATIONS:

            language = "English"


        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if "image" not in request.files:

            error = "Please select an image."

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                disease_info=disease_info,
                is_healthy=is_healthy,
                is_unknown=is_unknown,
                error=error,
                language=language,
                text=TRANSLATIONS[language]
            )


        file = request.files["image"]


        # ----------------------------------------------------
        # Check filename
        # ----------------------------------------------------

        if file.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                disease_info=disease_info,
                is_healthy=is_healthy,
                is_unknown=is_unknown,
                error=error,
                language=language,
                text=TRANSLATIONS[language]
            )


        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(image_path)


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = predict_image(
            image_path
        )


        # ====================================================
        # UNKNOWN / LOW CONFIDENCE
        # ====================================================

        if prediction is None:

            is_unknown = True

            print("\n" + "=" * 70)
            print("PREDICTION RESULT")
            print("=" * 70)

            print(
                "Result : Unable to confidently detect condition"
            )


        # ====================================================
        # VALID PREDICTION
        # ====================================================

        else:

    # ------------------------------------------------
    # Get disease information
    # ------------------------------------------------

           disease_info = DISEASE_INFO.get(
           prediction
    )


    # ------------------------------------------------
    # Save prediction to history
    # ------------------------------------------------

           if disease_info:

            condition_type = disease_info.get(
            "category",
            "Unknown"
        )

            save_prediction(
            prediction,
            condition_type
        )


    # ------------------------------------------------
    # Check healthy
    # ------------------------------------------------

            if disease_info:

                 is_healthy = (
                disease_info.get("category")
                == "Healthy"
            )


            # ------------------------------------------------
            # Check healthy
            # ------------------------------------------------

            if disease_info:

                is_healthy = (
                    disease_info.get("category")
                    == "Healthy"
                )


            # ------------------------------------------------
            # Terminal output
            # ------------------------------------------------

            print("\n" + "=" * 70)
            print("PREDICTION RESULT")
            print("=" * 70)

            print(
                "Predicted Condition :",
                prediction
            )

            print(
                "Selected Language   :",
                language
            )


            if is_healthy:

                print(
                    "Plant Status        : Healthy"
                )

            elif disease_info:

                print(
                    "Condition Type      :",
                    disease_info["category"]
                )

                print(
                    "Severity            :",
                    disease_info["severity"][language]
                )

            else:

                print(
                    "Information : Not available"
                )


    # ========================================================
    # SEND DATA TO HTML
    # ========================================================

    return render_template(

        "index.html",

        prediction=prediction,

        image_path=image_path,

        disease_info=disease_info,

        is_healthy=is_healthy,

        is_unknown=is_unknown,

        error=error,

        language=language,

        text=TRANSLATIONS[language]
    )

@app.route("/history")
def history():

    language = request.args.get("language", "English")

    if language not in TRANSLATIONS:
        language = "English"

    history_data = get_history()

    return render_template(
        "history.html",
        history=history_data,
        language=language,
        text=TRANSLATIONS[language],
        disease_info_map=DISEASE_INFO
    )



@app.route("/history/progress/<condition>")
def history_progress(condition):

    language = request.args.get("language", "English")

    if language not in TRANSLATIONS:
        language = "English"

    progress_data = get_condition_history(condition)

    disease_info = DISEASE_INFO.get(condition)

    return render_template(
        "progress.html",
        condition=condition,
        progress=progress_data,
        disease_info=disease_info,
        language=language,
        text=TRANSLATIONS[language]
    )

@app.route("/history/details/<condition>")
def history_details(condition):

    language = request.args.get("language", "English")

    if language not in TRANSLATIONS:
        language = "English"

    disease_info = DISEASE_INFO.get(condition)

    if not disease_info:
        return "Condition information not available."

    return render_template(
        "history_details.html",
        condition=condition,
        disease_info=disease_info,
        language=language,
        text=TRANSLATIONS[language]
    )

@app.route("/history/download")
def download_history():
    language = request.args.get("language", "English")

    if language not in TRANSLATIONS:
        language = "English"

    history = get_history()

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "prediction_history.csv"
    )

    with open(file_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "ID",
            "Date & Time",
            "Condition",
            "Type"
        ])

        for record in history:
            writer.writerow(record)

    return send_file(
        file_path,
        as_attachment=True,
        download_name="arecanut_prediction_history.csv"
    )

@app.route("/history/clear")
def clear_history_route():

    clear_history()

    language = request.args.get("language", "English")

    if language not in TRANSLATIONS:
        language = "English"

    return redirect(
        url_for("history", language=language)
    )

@app.route("/treatment/<condition>")
def treatment(condition):

    language = request.args.get("language", "English")

    if language not in TRANSLATIONS:
        language = "English"

    disease_info = DISEASE_INFO.get(condition)

    if not disease_info:
        return "Treatment information not available."

    return render_template(
        "treatment.html",
        condition=condition,
        disease_info=disease_info,
        language=language,
        text=TRANSLATIONS[language]
    )

# ============================================================
# 8. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("STARTING ARECANUT PLANT HEALTH DETECTION SYSTEM")
    print("=" * 70)

    app.run(
        debug=True
    )