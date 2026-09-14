import cv2
import numpy as np
import tensorflow as tf
import os


class TheftDetector:

    def __init__(self, model_path="shoplifting_detection_final.keras"):

        self.model = tf.keras.models.load_model(model_path)

        self.FRAME_COUNT = 32
        self.IMG_SIZE = 224


    def analyze_video(self, video_path, output_image_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open input video.")

        frames = []

        theft_detected = False
        highest_confidence = 0.0

        # Frame that will ultimately be displayed
        selected_frame = None

        current_label = "ANALYZING..."
        current_confidence = 0.0


        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Keep a copy of the original frame
            selected_frame = frame.copy()

            # ---------------------------------------------
            # PREPROCESS FRAME
            # ---------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            resized = cv2.resize(
                rgb_frame,
                (self.IMG_SIZE, self.IMG_SIZE)
            )

            normalized = (
                resized.astype(np.float32) / 255.0
            )

            frames.append(normalized)


            # ---------------------------------------------
            # PREDICT AFTER 32 FRAMES
            # ---------------------------------------------

            if len(frames) == self.FRAME_COUNT:

                input_data = np.array(frames)

                input_data = np.expand_dims(
                    input_data,
                    axis=0
                )

                prediction = self.model.predict(
                    input_data,
                    verbose=0
                )


                predicted_class = np.argmax(
                    prediction,
                    axis=1
                )[0]


                # -----------------------------------------
                # CLASS 1 = SHOPLIFTING
                # CLASS 0 = NORMAL
                # -----------------------------------------

                if predicted_class == 1:

                    confidence = (
                        float(prediction[0][1]) * 100
                    )

                    current_label = "THEFT DETECTED"
                    current_confidence = confidence

                    theft_detected = True

                    if confidence > highest_confidence:

                        highest_confidence = confidence

                        # IMPORTANT:
                        # Save the frame corresponding
                        # to the highest theft confidence
                        selected_frame = frame.copy()


                else:

                    confidence = (
                        float(prediction[0][0]) * 100
                    )

                    current_label = "NORMAL"
                    current_confidence = confidence

                    # If no theft has been detected,
                    # keep a representative normal frame
                    if not theft_detected:

                        selected_frame = frame.copy()


                # Sliding window
                frames.pop(0)


        cap.release()


        # ---------------------------------------------
        # SAFETY CHECK
        # ---------------------------------------------

        if selected_frame is None:

            raise ValueError(
                "No frames found in the video."
            )


        # ---------------------------------------------
        # DETERMINE FINAL RESULT
        # ---------------------------------------------

        if theft_detected:

            final_label = "THEFT DETECTED"
            final_confidence = highest_confidence

            text_color = (255, 255, 255)
            box_color = (0, 0, 200)

        else:

            final_label = "NORMAL"

            final_confidence = current_confidence

            text_color = (255, 255, 255)
            box_color = (0, 140, 0)


        # ---------------------------------------------
        # WRITE RESULT ON SELECTED FRAME
        # ---------------------------------------------

        cv2.rectangle(
            selected_frame,
            (20, 20),
            (550, 115),
            box_color,
            -1
        )


        cv2.putText(
            selected_frame,
            final_label,
            (40, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            3,
            cv2.LINE_AA
        )


        confidence_text = (
            f"Confidence: {final_confidence:.1f}%"
        )


        cv2.putText(
            selected_frame,
            confidence_text,
            (40, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_color,
            2,
            cv2.LINE_AA
        )


        # ---------------------------------------------
        # SAVE ONLY ONE IMAGE
        # ---------------------------------------------

        cv2.imwrite(
            output_image_path,
            selected_frame
        )


        return (
            theft_detected,
            final_confidence,
            output_image_path
        )