
import os
import uuid

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, send_from_directory


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "shoplifting_detection_final.keras"
)

FRAME_COUNT = 32
IMG_SIZE = 224

NORMAL_CLASS = 0
SHOPLIFTING_CLASS = 1


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# SERVE RESULT IMAGE
# ============================================================

@app.route("/outputs/<filename>")
def output_image(filename):
    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


# ============================================================
# UPLOAD + PREDICT
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if "video" not in request.files:
        return redirect(url_for("index"))

    video = request.files["video"]

    if video.filename == "":
        return redirect(url_for("index"))

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    original_name = video.filename
    extension = os.path.splitext(original_name)[1].lower()

    if extension not in [".mp4", ".avi", ".mov", ".mkv"]:
        return "Unsupported video format."

    video_filename = f"video_{unique_id}{extension}"

    video_path = os.path.join(
        UPLOAD_FOLDER,
        video_filename
    )

    video.save(video_path)

    print("Uploaded video:", video_path)

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Could not open uploaded video."

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print("FPS:", fps)
    print("Total frames:", total_frames)

    # --------------------------------------------------------
    # Frame buffer
    # --------------------------------------------------------

    frames = []

    representative_frame = None

    prediction_text = "NORMAL"
    confidence = 0.0

    prediction_made = False

    # --------------------------------------------------------
    # Read video frames
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Keep original frame for webpage display
        display_frame = frame.copy()

        # ----------------------------------------------------
        # Convert BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Resize to 224 x 224
        # ----------------------------------------------------

        resized = cv2.resize(
            rgb_frame,
            (IMG_SIZE, IMG_SIZE)
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        resized = resized.astype(
            np.float32
        ) / 255.0

        # ----------------------------------------------------
        # Add to 32-frame buffer
        # ----------------------------------------------------

        frames.append(resized)

        # ----------------------------------------------------
        # Predict after collecting 32 frames
        # ----------------------------------------------------

        if len(frames) == FRAME_COUNT:

            input_data = np.array(
                frames,
                dtype=np.float32
            )

            # Add batch dimension
            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            print(
                "Prediction input shape:",
                input_data.shape
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # Same approach as your notebook
            # ------------------------------------------------

            prediction = model.predict(
                input_data,
                verbose=0
            )

            binary_prediction = int(
                np.argmax(prediction, axis=1)[0]
            )

            # ------------------------------------------------
            # Confidence
            #
            # Following the logic from your notebook:
            # probability = prediction[0][0]
            # ------------------------------------------------

            probability = float(
                prediction[0][0]
            )

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            if binary_prediction == SHOPLIFTING_CLASS:

                prediction_text = "THEFT DETECTED"

                confidence = probability * 100

            else:

                prediction_text = "NORMAL"

                confidence = (1 - probability) * 100

            prediction_made = True

            # ------------------------------------------------
            # IMPORTANT:
            # Keep an actual frame from the uploaded video.
            #
            # This is the frame corresponding to the current
            # prediction window.
            # ------------------------------------------------

            representative_frame = display_frame.copy()

            # ------------------------------------------------
            # Sliding window
            # ------------------------------------------------

            frames.pop(0)

            # ------------------------------------------------
            # We only need ONE frame for the webpage.
            # Stop after the first valid prediction.
            # ------------------------------------------------

            break

    cap.release()

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if not prediction_made or representative_frame is None:

        return (
            "The uploaded video does not contain enough frames. "
            "Please upload a video containing at least 32 frames."
        )

    # ========================================================
    # ADD RESULT TO FRAME
    # ========================================================

    if prediction_text == "THEFT DETECTED":

        text = f"THEFT DETECTED  |  {confidence:.1f}%"

        text_color = (0, 0, 255)

    else:

        text = f"NORMAL  |  {confidence:.1f}%"

        text_color = (0, 180, 0)

    # --------------------------------------------------------
    # Add dark background strip
    # --------------------------------------------------------

    cv2.rectangle(
        representative_frame,
        (0, 0),
        (representative_frame.shape[1], 75),
        (20, 20, 20),
        -1
    )

    # --------------------------------------------------------
    # Add prediction text
    # --------------------------------------------------------

    cv2.putText(
        representative_frame,
        text,
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        text_color,
        3,
        cv2.LINE_AA
    )

    # ========================================================
    # SAVE ONLY ONE IMAGE
    # ========================================================

    image_filename = (
        f"detection_{unique_id}.jpg"
    )

    image_path = os.path.join(
        OUTPUT_FOLDER,
        image_filename
    )

    success = cv2.imwrite(
        image_path,
        representative_frame
    )

    if not success:
        return "Could not save detection frame."

    print("Detection frame saved:", image_path)

    # ========================================================
    # RESULT PAGE
    # ========================================================

    return render_template(
        "result.html",
        result=prediction_text,
        confidence=f"{confidence:.1f}",
        image_file=image_filename
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

