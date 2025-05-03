import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from ultralytics import YOLO
import os
import re # Import regex for better character filtering

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Saudi License Plate OCR")
st.title("🇸🇦 Saudi License Plate Character Recognition")
st.write("Upload an image of a Saudi license plate to detect and recognize the characters using YOLO and Tesseract OCR.")

# --- Model Loading ---
MODEL_PATH = "best.pt"
FALLBACK_MODEL_PATH = "yolov8n.pt" # Base model if trained one not found
# Reverted to original font path as requested, ensure path is correct
ARABIC_FONT_PATH = "/usr/share/fonts/truetype/fonts-arabeyes/ae_Arab.ttf" # Reverted to ae_Arab
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
    font_found = False
    try:
        if os.path.exists(ARABIC_FONT_PATH):
            font_path_to_use = ARABIC_FONT_PATH
            font = ImageFont.truetype(font_path_to_use, font_size)
            st.success(f"Attempting to load Arabic font: {ARABIC_FONT_PATH}") # Changed message slightly
            font_found = True
            return font
        else:
            st.warning(f"Arabic font not found at {ARABIC_FONT_PATH}. Trying fallback.")
    except Exception as e:
        st.warning(f"Error loading Arabic font {ARABIC_FONT_PATH}: {e}. Trying fallback.")

    try:
        if os.path.exists(FALLBACK_FONT_PATH):
            font_path_to_use = FALLBACK_FONT_PATH
            font = ImageFont.truetype(font_path_to_use, font_size)
            st.info(f"Using fallback font: {FALLBACK_FONT_PATH}")
            font_found = True
            return font
        else:
            st.error(f"Fallback font not found at {FALLBACK_FONT_PATH}. Text rendering might fail.")
    except Exception as e:
        st.error(f"Error loading fallback font {FALLBACK_FONT_PATH}: {e}. Text rendering might fail.")

    if not font_found:
        st.error("No suitable font found. Cannot render text on image.")
        return None

# Load font with a specific size for drawing on image
font_for_drawing = load_font(font_size=25)

# --- Helper Function for Character Classification ---
def classify_char(char):
    """Classify character as Arabic Letter, Digit, or Other."""
    if not char or char == "?":
        return "Other"
    # Check for Arabic letters
    if re.search(r"^[\u0600-\u06FF]+$", char):
        return "Arabic Letter"
    # Check for digits (Arabic-Indic or European)
    if re.search(r"^[0-9\u0660-\u0669]+$", char):
        return "Digit"
    # Check for English letters (might be misclassified Arabic)
    if re.search(r"^[a-zA-Z]+$", char):
         return "English Letter" # Treat as potential letter
    return "Other"

# --- OCR Function ---
def apply_ocr_on_yolo_boxes(image_np):
    """Applies YOLO detection, Tesseract OCR, and draws text on the image.
       Returns a list of (box_coords, text, type) pairs and the annotated image.
    """
    box_text_type_pairs = [] # Store (box_coords, text, type) pairs
    annotated_image = image_np.copy()
    results = model(image_np)

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        h, w = image_np.shape[:2]
        try:
            # Sort boxes primarily by y-coordinate (top to bottom), then x-coordinate (left to right)
            sorted_boxes = sorted(results[0].boxes, key=lambda b: (b.xyxy[0][1].item(), b.xyxy[0][0].item()))
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
            extracted_text = "?"
            char_type = "Other"

            try:
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                # Apply adaptive thresholding
                thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                # OCR Config: PSM 10 assumes a single character
                custom_config_ara = r"--oem 3 --psm 10 -l ara"
                custom_config_eng = r"--oem 3 --psm 10 -l eng+ara" # Try eng+ara for digits

                # Try Arabic first
                ocr_text_ara = pytesseract.image_to_string(thresholded_image, config=custom_config_ara).strip()
                ocr_text_ara = re.sub(r"[^\u0600-\u06FF0-9\u0660-\u0669a-zA-Z]", "", ocr_text_ara)

                # Try Eng/Digits second
                ocr_text_eng = pytesseract.image_to_string(thresholded_image, config=custom_config_eng).strip()
                ocr_text_eng = re.sub(r"[^\u0600-\u06FF0-9\u0660-\u0669a-zA-Z]", "", ocr_text_eng)

                # Prioritize Arabic letters if found
                if ocr_text_ara and classify_char(ocr_text_ara) == "Arabic Letter":
                    extracted_text = ocr_text_ara
                # Else prioritize digits found by either
                elif ocr_text_eng and classify_char(ocr_text_eng) == "Digit":
                     extracted_text = ocr_text_eng
                elif ocr_text_ara and classify_char(ocr_text_ara) == "Digit":
                     extracted_text = ocr_text_ara
                # Fallback to English letters or whatever was found
                elif ocr_text_eng:
                    extracted_text = ocr_text_eng
                elif ocr_text_ara:
                    extracted_text = ocr_text_ara
                else:
                    extracted_text = "?"

                # Ensure only one character is stored if PSM 10 is effective
                if len(extracted_text) > 1:
                    digits_found = re.findall(r"[0-9\u0660-\u0669]", extracted_text)
                    if digits_found:
                        extracted_text = digits_found[0]
                    else:
                        extracted_text = extracted_text[0]
                elif len(extracted_text) == 0:
                    extracted_text = "?"

                char_type = classify_char(extracted_text)

            except pytesseract.TesseractNotFoundError:
                st.error("Tesseract is not installed or not in your PATH.")
                return "Error: Tesseract not found", image_np
            except cv2.error as e:
                st.warning(f"OpenCV error during preprocessing box ({x1},{y1},{x2},{y2}): {e}")
            except Exception as e:
                st.warning(f"Error during OCR/preprocessing for box ({x1},{y1},{x2},{y2}): {e}")

            box_text_type_pairs.append(((x1, y1, x2, y2), extracted_text, char_type))

        # Second pass: Draw boxes and text using PIL
        try:
            annotated_image_yolo_boxes = results[0].plot()
            pil_image = Image.fromarray(cv2.cvtColor(annotated_image_yolo_boxes, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            if font_for_drawing:
                for box, text, _ in box_text_type_pairs:
                    x1, y1, x2, y2 = box
                    text_x = x1
                    text_y = max(0, y1 - font_for_drawing.size - 2)
                    try:
                        if hasattr(font_for_drawing, "getbbox"):
                            bbox = draw.textbbox((text_x, text_y), text, font=font_for_drawing)
                        else:
                            text_width, text_height = draw.textsize(text, font=font_for_drawing)
                            bbox = (text_x, text_y, text_x + text_width, text_y + text_height)
                        bg_bbox = (bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2)
                        draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
                    except Exception as e:
                        st.warning(f"Could not draw text background: {e}")

                    draw.text((text_x, text_y), text, fill=(255, 255, 0), font=font_for_drawing)
            else:
                st.warning("Font not loaded, cannot draw text on image.")

            annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            st.error(f"Error drawing text or plotting YOLO results: {e}")
            annotated_image = results[0].plot() if results and results[0] else image_np

        return box_text_type_pairs, annotated_image

    else:
        st.warning("No license plate characters detected by YOLO.")
        return [], image_np # Return empty list and original image

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Choose a license plate image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    display_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.write("Processing...")
        # Perform OCR and get annotated image and box/text/type pairs
        ocr_result, annotated_display_image = apply_ocr_on_yolo_boxes(opencv_image)

        # Display annotated image
        annotated_display_image_rgb = cv2.cvtColor(annotated_display_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_display_image_rgb, caption="Processed Image with Detections & Text", use_column_width=True)

        st.subheader("Extracted Plate Text:")

        if isinstance(ocr_result, str): # Handle error string from OCR function
            st.code(ocr_result, language=None)
        elif ocr_result: # Should be list of tuples
            letters = []
            numbers = []
            raw_sequence = []

            for _, text, char_type in ocr_result:
                raw_sequence.append(text) # Keep track of original detected sequence
                if char_type == "Arabic Letter" or char_type == "English Letter":
                    letters.append(text)
                elif char_type == "Digit":
                    numbers.append(text)
                else: # Handle "Other" or "?"
                    pass # Discard placeholders from formatted output

            # Format for Saudi Plate (Letters RTL, Numbers LTR)
            # Reverse the detected letter sequence for correct RTL display
            formatted_letters = " ".join(letters[::-1])
            formatted_numbers = " ".join(numbers)

            # Display using HTML for directional control
            # Reverted font family name in HTML
            plate_html = f"""
            <div style="border: 1px solid #ccc; padding: 10px; margin-top: 10px; background-color: #f0f0f0; text-align: center;">
                <span style="font-size: 2em; direction: rtl; unicode-bidi: bidi-override; display: inline-block; margin-right: 15px; font-family: 'ae_Arab', 'DejaVu Sans', sans-serif; font-weight: bold; color: #333;">
                    {formatted_letters if formatted_letters else '[Letters?]'}
                </span>
                <span style="font-size: 2em; direction: ltr; display: inline-block; font-family: 'DejaVu Sans', sans-serif; font-weight: bold; color: #333;">
                    {formatted_numbers if formatted_numbers else '[Numbers?]'}
                </span>
            </div>
            """
            st.markdown("**Formatted Plate:**")
            st.markdown(plate_html, unsafe_allow_html=True)
            st.markdown("*Note: Assumes standard Saudi plate format (Letters first, then Numbers). RTL applied to letters.*")

            # Show the raw sequence for debugging
            st.markdown("**Raw Detected Sequence (Left-to-Right):**")
            st.code(" ".join(raw_sequence), language=None)

        else:
             st.code("No characters detected or extracted.", language=None)

        # Check Tesseract languages
        try:
            langs = pytesseract.get_languages(config="")
            st.info(f"Available Tesseract languages: {langs}")
            if "ara" not in langs:
                st.warning("Arabic language data ('ara') for Tesseract might not be installed. OCR accuracy for Arabic may be low. You might need to install 'tesseract-ocr-ara'.")
            if "eng" not in langs:
                 st.warning("English language data ('eng') for Tesseract might not be installed.")
        except pytesseract.TesseractNotFoundError:
             pass # Error already handled
        except Exception as e:
             st.warning(f"Could not check Tesseract languages: {e}")

