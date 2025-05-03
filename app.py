import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from ultralytics import YOLO
import os

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Saudi License Plate OCR")
st.title("🇸🇦 Saudi License Plate Character Recognition")
st.write("Upload an image of a Saudi license plate to detect and recognize the characters using YOLO and Tesseract OCR.")

# --- Model Loading ---
# IMPORTANT: Assuming the trained YOLO model weights are saved here.
# The original notebook trained a model named 'yolo_characters_custom'.
# If this path is incorrect, please provide the correct path to 'best.pt'.
MODEL_PATH = "best.pt"
FALLBACK_MODEL_PATH = "yolov8n.pt" # Base model if trained one not found

@st.cache_resource # Cache the model loading
def load_yolo_model(model_path):
    """Loads the YOLO model."""
    if os.path.exists(model_path):
        try:
            model = YOLO(model_path)
            st.success(f"Loaded trained model: {model_path}")
            return model
        except Exception as e:
            st.error(f"Error loading trained model from {model_path}: {e}")
            st.warning(f"Falling back to base model: {FALLBACK_MODEL_PATH}")
            return YOLO(FALLBACK_MODEL_PATH)
    else:
        st.warning(f"Trained model not found at {model_path}. Falling back to base model: {FALLBACK_MODEL_PATH}")
        return YOLO(FALLBACK_MODEL_PATH)

model = load_yolo_model(MODEL_PATH)

# --- OCR Function ---
def apply_ocr_on_yolo_boxes(image_np):
    """Applies YOLO detection and Tesseract OCR on the image."""
    texts = []
    annotated_image = image_np.copy()

    # Use YOLO to detect objects in the image
    results = model(image_np)

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        # Get image dimensions
        h, w = image_np.shape[:2]

        # Sort boxes from left to right based on the x-coordinate of the top-left corner
        # Accessing box coordinates might differ slightly based on ultralytics version, adjust if needed
        try:
            # Assuming results[0].boxes contains box data including xyxy
            sorted_boxes = sorted(results[0].boxes, key=lambda b: b.xyxy[0][0].item())
        except Exception as e:
            st.error(f"Error sorting boxes: {e}. Using unsorted boxes.")
            sorted_boxes = results[0].boxes

        for box in sorted_boxes:
            # Extract coordinates
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
            except Exception as e:
                st.error(f"Error extracting box coordinates: {e}")
                continue

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            # Check if the box has valid dimensions
            if x1 >= x2 or y1 >= y2:
                st.warning(f"Skipping invalid box with coordinates: ({x1}, {y1}, {x2}, {y2})")
                continue

            # Crop character region
            cropped_image = image_np[y1:y2, x1:x2]

            # Preprocess for OCR
            try:
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                # Apply thresholding (Otsu's method)
                _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                # Optional: Add padding if needed for better OCR
                # thresholded_image = cv2.copyMakeBorder(thresholded_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0,0,0])
            except cv2.error as e:
                st.warning(f"OpenCV error during preprocessing box ({x1},{y1},{x2},{y2}): {e}. Skipping box.")
                continue
            except Exception as e:
                st.warning(f"Error during preprocessing box ({x1},{y1},{x2},{y2}): {e}. Skipping box.")
                continue

            # OCR with Tesseract
            try:
                # --psm 7: Treat the image as a single text line.
                # --oem 3: Default OCR Engine Mode
                # -l ara: Specify Arabic language (adjust if needed, e.g., 'eng' for English)
                # Check if Arabic language data is installed for Tesseract
                custom_config = r'--oem 3 --psm 7 -l ara'
                extracted_text = pytesseract.image_to_string(thresholded_image, config=custom_config).strip()

                # Basic filtering (remove non-alphanumeric if needed, depends on expected characters)
                # extracted_text = ''.join(filter(str.isalnum, extracted_text))

                if extracted_text:
                    texts.append(extracted_text)
                else:
                     # Try English if Arabic fails or is empty
                     custom_config_eng = r'--oem 3 --psm 7 -l eng'
                     extracted_text_eng = pytesseract.image_to_string(thresholded_image, config=custom_config_eng).strip()
                     if extracted_text_eng:
                         texts.append(extracted_text_eng)

            except pytesseract.TesseractNotFoundError:
                st.error("Tesseract is not installed or not in your PATH. Please install Tesseract.")
                return "Error", image_np # Return early
            except Exception as e:
                st.warning(f"Error during OCR for box ({x1},{y1},{x2},{y2}): {e}")
                texts.append("?") # Placeholder for error

        # Plot YOLO results on the image (bounding boxes and labels)
        try:
            annotated_image = results[0].plot() # This returns a numpy array (BGR)
        except Exception as e:
            st.error(f"Error plotting YOLO results: {e}")
            # Keep the original image if plotting fails

        final_text = ' '.join(texts) # Join characters with space
        return final_text, annotated_image

    else:
        st.warning("No license plate characters detected by YOLO.")
        return "No characters detected", image_np

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Choose a license plate image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # Convert BGR to RGB for display
    display_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.image(display_image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.write("Processing...")
        # Perform OCR
        extracted_text, annotated_display_image = apply_ocr_on_yolo_boxes(opencv_image)

        # Display annotated image (convert BGR from plot() to RGB)
        annotated_display_image_rgb = cv2.cvtColor(annotated_display_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_display_image_rgb, caption='Processed Image with Detections', use_column_width=True)

        st.subheader("Extracted Text:")
        st.code(extracted_text, language=None)

        # Check Tesseract languages
        try:
            langs = pytesseract.get_languages(config='')
            st.info(f"Available Tesseract languages: {langs}")
            if 'ara' not in langs:
                st.warning("Arabic language data ('ara') for Tesseract might not be installed. OCR accuracy for Arabic may be low. You might need to install 'tesseract-ocr-ara'.")
        except pytesseract.TesseractNotFoundError:
             pass # Error already handled in OCR function
        except Exception as e:
             st.warning(f"Could not check Tesseract languages: {e}")

