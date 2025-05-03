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
    """Applies YOLO detection and Tesseract OCR on the image with improved sorting."""
    detected_chars = [] # Store tuples of (box_coords, text)
    annotated_image = image_np.copy()

    # Use YOLO to detect objects in the image
    results = model(image_np)

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        # Get image dimensions
        h, w = image_np.shape[:2]

        boxes_data = [] # Store box data for sorting: (x1, y1, x2, y2, center_y)
        for box in results[0].boxes:
             try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, w))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h))
                y2 = max(0, min(y2, h))
                # Check if the box has valid dimensions
                if x1 < x2 and y1 < y2:
                    center_y = (y1 + y2) / 2
                    boxes_data.append(((x1, y1, x2, y2), center_y))
                else:
                    st.warning(f"Skipping invalid box with coordinates: ({x1}, {y1}, {x2}, {y2})")
             except Exception as e:
                st.error(f"Error extracting box coordinates: {e}")
                continue

        if not boxes_data:
            st.warning("No valid character boxes found after filtering.")
            return "No characters detected", image_np

        # --- Improved Sorting Logic ---
        # Sort primarily by vertical position (center_y), then by horizontal position (x1)
        # Group boxes by approximate row based on vertical overlap or proximity
        boxes_data.sort(key=lambda item: item[1]) # Initial sort by center_y

        rows = []
        if boxes_data:
            current_row = [boxes_data[0]]
            last_box_coords, last_center_y = boxes_data[0]

            for i in range(1, len(boxes_data)):
                box_coords, center_y = boxes_data[i]
                # Check if the current box vertically overlaps significantly or is close to the last box in the current row
                # Use a threshold based on box height (e.g., half the height)
                box_height = box_coords[3] - box_coords[1]
                vertical_threshold = box_height * 0.5 # Adjust this threshold as needed

                if abs(center_y - last_center_y) < vertical_threshold:
                    current_row.append(boxes_data[i])
                else:
                    # Sort the completed row horizontally
                    current_row.sort(key=lambda item: item[0][0]) # Sort by x1
                    rows.append(current_row)
                    current_row = [boxes_data[i]] # Start a new row

                last_box_coords, last_center_y = boxes_data[i] # Update last box info

            # Add the last row after sorting it horizontally
            current_row.sort(key=lambda item: item[0][0])
            rows.append(current_row)

        # --- OCR Processing (Iterate through sorted rows/boxes) ---
        processed_texts = []
        for row in rows:
            row_texts = []
            for box_data in row:
                (x1, y1, x2, y2), _ = box_data
                # Crop character region
                cropped_image = image_np[y1:y2, x1:x2]

                # Preprocess for OCR
                try:
                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                except cv2.error as e:
                    st.warning(f"OpenCV error during preprocessing box ({x1},{y1},{x2},{y2}): {e}. Skipping box.")
                    row_texts.append("?")
                    continue
                except Exception as e:
                    st.warning(f"Error during preprocessing box ({x1},{y1},{x2},{y2}): {e}. Skipping box.")
                    row_texts.append("?")
                    continue

                # OCR with Tesseract
                extracted_text = "?" # Default to placeholder
                try:
                    custom_config_ara = r'--oem 3 --psm 7 -l ara'
                    text_ara = pytesseract.image_to_string(thresholded_image, config=custom_config_ara).strip()

                    custom_config_eng = r'--oem 3 --psm 7 -l eng'
                    text_eng = pytesseract.image_to_string(thresholded_image, config=custom_config_eng).strip()

                    # Prioritize Arabic if found, otherwise use English if found
                    if text_ara:
                        extracted_text = text_ara
                    elif text_eng:
                         # Basic filtering for English (allow only uppercase letters and digits)
                         filtered_eng = ''.join(filter(lambda char: char.isalnum() and char.isupper() or char.isdigit(), text_eng))
                         if filtered_eng:
                             extracted_text = filtered_eng

                except pytesseract.TesseractNotFoundError:
                    st.error("Tesseract is not installed or not in your PATH. Please install Tesseract.")
                    return "Error: Tesseract not found", image_np # Return early
                except Exception as e:
                    st.warning(f"Error during OCR for box ({x1},{y1},{x2},{y2}): {e}")
                    # Keep extracted_text as "?"

                row_texts.append(extracted_text)
            processed_texts.append("".join(row_texts)) # Join characters within a row without space

        # Plot YOLO results on the image (bounding boxes and labels)
        try:
            annotated_image = results[0].plot() # This returns a numpy array (BGR)
        except Exception as e:
            st.error(f"Error plotting YOLO results: {e}")
            # Keep the original image if plotting fails

        # Join rows with a space or newline depending on desired format
        final_text = ' '.join(processed_texts)
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
        # Display text with right-to-left direction if it contains Arabic characters
        # Basic check for Arabic characters
        contains_arabic = any('\u0600' <= char <= '\u06FF' for char in extracted_text)
        if contains_arabic:
            st.markdown(f'<div dir="rtl">{extracted_text}</div>', unsafe_allow_html=True)
        else:
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
