import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
# Attempt to find a suitable Arabic font, fallback to DejaVuSans
ARABIC_FONT_PATH = "/usr/share/fonts/truetype/arabeyes/ae_Arab.ttf" # From fonts-arabeyes package
FALLBACK_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Default on many Linux systems

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

# --- Font Loading ---
@st.cache_resource
def load_font(font_size=20):
    """Loads the font for drawing text."""
    font_path_to_use = None
    if os.path.exists(ARABIC_FONT_PATH):
        font_path_to_use = ARABIC_FONT_PATH
        st.info(f"Using Arabic font: {ARABIC_FONT_PATH}")
    elif os.path.exists(FALLBACK_FONT_PATH):
        font_path_to_use = FALLBACK_FONT_PATH
        st.warning(f"Arabic font not found at {ARABIC_FONT_PATH}. Using fallback font: {FALLBACK_FONT_PATH}")
    else:
        st.error("Neither Arabic nor fallback font found. Text rendering might fail or look incorrect.")
        return None

    try:
        return ImageFont.truetype(font_path_to_use, font_size)
    except Exception as e:
        st.error(f"Error loading font {font_path_to_use}: {e}")
        return None

font = load_font(font_size=25) # Adjust size as needed

# --- OCR Function ---
def apply_ocr_on_yolo_boxes(image_np):
    """Applies YOLO detection, Tesseract OCR, and draws text on the image."""
    texts = []
    box_text_pairs = [] # Store (box_coords, text) pairs
    annotated_image = image_np.copy()

    # Use YOLO to detect objects in the image
    results = model(image_np)

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        # Get image dimensions
        h, w = image_np.shape[:2]

        # Sort boxes from left to right
        try:
            sorted_boxes = sorted(results[0].boxes, key=lambda b: b.xyxy[0][0].item())
        except Exception as e:
            st.error(f"Error sorting boxes: {e}. Using unsorted boxes.")
            sorted_boxes = results[0].boxes

        # First pass: OCR each box
        for box in sorted_boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
            except Exception as e:
                st.error(f"Error extracting box coordinates: {e}")
                continue

            x1, y1, x2, y2 = max(0, min(x1, w)), max(0, min(y1, h)), max(0, min(x2, w)), max(0, min(y2, h))

            if x1 >= x2 or y1 >= y2:
                st.warning(f"Skipping invalid box: ({x1}, {y1}, {x2}, {y2})")
                continue

            cropped_image = image_np[y1:y2, x1:x2]
            extracted_text = "?" # Default placeholder

            try:
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # OCR with Tesseract (Arabic first, then English fallback)
                custom_config_ara = r'--oem 3 --psm 7 -l ara'
                ocr_text_ara = pytesseract.image_to_string(thresholded_image, config=custom_config_ara).strip()

                if ocr_text_ara and any(c.isalnum() for c in ocr_text_ara): # Check if not just symbols
                    extracted_text = ocr_text_ara
                else:
                     custom_config_eng = r'--oem 3 --psm 7 -l eng'
                     ocr_text_eng = pytesseract.image_to_string(thresholded_image, config=custom_config_eng).strip()
                     if ocr_text_eng and any(c.isalnum() for c in ocr_text_eng):
                         extracted_text = ocr_text_eng

            except pytesseract.TesseractNotFoundError:
                st.error("Tesseract is not installed or not in your PATH.")
                return "Error: Tesseract not found", image_np # Return early
            except cv2.error as e:
                st.warning(f"OpenCV error during preprocessing box ({x1},{y1},{x2},{y2}): {e}")
            except Exception as e:
                st.warning(f"Error during OCR/preprocessing for box ({x1},{y1},{x2},{y2}): {e}")

            texts.append(extracted_text)
            box_text_pairs.append(((x1, y1, x2, y2), extracted_text))

        # Second pass: Draw boxes and text using PIL for better font support
        try:
            # Get the base image with YOLO boxes plotted (returns BGR numpy array)
            annotated_image_yolo_boxes = results[0].plot()
            # Convert to PIL RGB Image
            pil_image = Image.fromarray(cv2.cvtColor(annotated_image_yolo_boxes, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            if font:
                for box, text in box_text_pairs:
                    x1, y1, x2, y2 = box
                    # Position text slightly above the box
                    text_x = x1
                    text_y = max(0, y1 - font.size - 2) # Position above, with small padding
                    # Draw background rectangle for text for better visibility
                    try:
                        # Use getbbox if available (Pillow >= 9.2.0), otherwise fallback
                        if hasattr(font, 'getbbox'):
                            bbox = draw.textbbox((text_x, text_y), text, font=font)
                        else:
                            # Fallback for older Pillow versions (less accurate)
                            text_width, text_height = draw.textsize(text, font=font)
                            bbox = (text_x, text_y, text_x + text_width, text_y + text_height)
                        draw.rectangle(bbox, fill=(0, 0, 0, 180)) # Semi-transparent black background
                    except Exception as e:
                        st.warning(f"Could not draw text background: {e}")
                        bbox = (text_x, text_y, text_x + 50, text_y + font.size) # Placeholder bbox

                    # Draw the text
                    draw.text((text_x, text_y), text, fill=(255, 255, 0), font=font) # Yellow text
            else:
                st.warning("Font not loaded, cannot draw text on image.")

            # Convert back to OpenCV BGR format for consistency
            annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            st.error(f"Error drawing text or plotting YOLO results: {e}")
            annotated_image = results[0].plot() if results and results[0] else image_np # Fallback to YOLO plot or original

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
        # Perform OCR and get annotated image
        extracted_text, annotated_display_image = apply_ocr_on_yolo_boxes(opencv_image)

        # Display annotated image (it's already BGR from our function, convert to RGB for st.image)
        annotated_display_image_rgb = cv2.cvtColor(annotated_display_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_display_image_rgb, caption='Processed Image with Detections & Text', use_column_width=True)

        st.subheader("Extracted Text (Concatenated):")
        st.code(extracted_text, language=None)

        # Check Tesseract languages
        try:
            langs = pytesseract.get_languages(config='')
            st.info(f"Available Tesseract languages: {langs}")
            if 'ara' not in langs:
                st.warning("Arabic language data ('ara') for Tesseract might not be installed. OCR accuracy for Arabic may be low. You might need to install 'tesseract-ocr-ara'.")
            if 'eng' not in langs:
                 st.warning("English language data ('eng') for Tesseract might not be installed.")
        except pytesseract.TesseractNotFoundError:
             pass # Error already handled in OCR function
        except Exception as e:
             st.warning(f"Could not check Tesseract languages: {e}")


