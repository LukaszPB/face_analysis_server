from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import dlib
import logging
import sys

# Ładowanie modelu do detekcji punktów twarzy
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

def predict_eye_color(landmarks, image_hsv):
    # Pobranie współrzędnych oczu (lewego i prawego)
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(43, 48) if i != 45])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(37, 42) if i != 39])

    # Tworzenie maski dla tęczówki
    mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [left_eye], 255)
    cv2.fillPoly(mask, [right_eye], 255)

    mask[(image_hsv[:, :, 2] > 140) | (image_hsv[:, :, 2] < 60)] = 0

    # Pobranie kolorów tęczówki
    eye_region = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    eye_pixels = eye_region[np.where(mask == 255)]
    avg_color = np.mean(eye_pixels, axis=0)  # Średnia wartość koloru BGR

    #Określenie koloru na podstawie wartości RGB
    def classify_eye_color_hsv(hsv):
        h, s, v = hsv
        if s < 40 or (s < 50 and h < 30):
            return "gray"
        elif h < 30 and s > 110:
            return "hazel"
        elif h > 90 and s < 100:
            return "blue"
        elif (32 < h < 45 and s < 65) or (h < 32 and s < 85 and v > 100):
            return "green"
        elif h < 65 and s > 75 and v < 100:
            return "brown"
        else:
            return "-"
        
    return classify_eye_color_hsv(avg_color)
        
def predict_hair_color(landmarks, image_hsv):
    hair_region = image_hsv[0:landmarks.part(74).y, max(landmarks.part(0).x-100,0):landmarks.part(16).x+100]

    # Wykrywanie obszaru włosów za pomocą operacji analizy obazów 
    edges = cv2.Canny(hair_region, 50, 75)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(hair_region[:, :, 0])
    cv2.drawContours(filled, contours, -1, (255), thickness=cv2.FILLED)

    # Usunięcie pikseli o skrajnie innym kolorze HSV
    hsv_values = hair_region[filled == 255]
    mean_hsv = np.mean(hsv_values, axis=0)

    sat_tolerance = 80
    val_tolerance = 60

    lower_bound = np.array([
        0,
        mean_hsv[1] - sat_tolerance,
        mean_hsv[2] - val_tolerance
    ])
    upper_bound = np.array([
        255,
        mean_hsv[1] + sat_tolerance,
        mean_hsv[2] + val_tolerance
    ])

    filtered_mask = cv2.inRange(hair_region, lower_bound, upper_bound)
    filtered_mask = cv2.resize(filtered_mask, (filled.shape[1], filled.shape[0]))
    cleaned_mask = cv2.bitwise_and(filled, filtered_mask)

    final_hsv_values = hair_region[cleaned_mask == 255] 
    final_mean_hsv = np.mean(final_hsv_values, axis=0)

    print(final_mean_hsv)

    h,s,v = final_mean_hsv
    if (s >= 120 or v >= 120 or (s >= 100 and v >= 100)) and s >= 60 and v >= 60 and h >= 10:
        return "blonde"
    elif h >= 25:
        return "black"
    elif h >= 10:
        return "brown"
    else:
        return "-"

def predict_face_shape(landmarks):
    x_min, x_max, y_min, y_max = 9999,0,9999,0
    for n in range(0, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        x_min=min(x_min,x)
        x_max=max(x_max,x)
        y_min=min(y_min,y)
        y_max=max(y_max,y)

    height_width = (y_max-y_min)/(x_max-x_min)
    chin_proportion = (landmarks.part(8).y-landmarks.part(5).y)/(landmarks.part(10).x-landmarks.part(6).x)
    forehead_jaw = (landmarks.part(79).x-landmarks.part(75).x)/(landmarks.part(12).x-landmarks.part(4).x)

    print(chin_proportion)

    if height_width > 1.28:
        if chin_proportion < 0.35:
            return "rectangle"
        else:
            return "triangle"
    else:
        if chin_proportion < 0.35:
            if 1.08 > forehead_jaw > 0.96:
                 return "pear"
            else:
                return "square"
        elif chin_proportion < 0.40:
            return "round"
        else:
            return "oval"

def predict_skin_tone(eye_color, hair_color):
    # if(eye_color == "")
    return "summer"

app = Flask(__name__)
CORS(app)

# Konfiguracja loggera raz, przy starcie
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        logger.info("Received request with files: %s", request.files.keys())

        if 'image' not in request.files:
            logger.warning("No image file provided")
            return {"error": "No image file provided"}, 400

        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")
        logger.info("Image opened successfully")

        image_np = np.array(image)
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Wykrywanie twarzy
        faces = detector(image_gray)

        if len(faces) == 0:
            logger.warning("Face not detected")
            return {"error": "Face not detected"}, 400
        if len(faces) > 1:
            logger.warning("Too many faces detected: %d", len(faces))
            return {"error": f"Too many faces detected: {len(faces)}"}, 400

        landmarks = predictor(image_gray, faces[0])

        eye_color = predict_eye_color(landmarks, image_hsv)
        hair_color = predict_hair_color(landmarks, image_hsv)
        face_shape = predict_face_shape(landmarks)
        skin_tone = predict_skin_tone()

        traits = {
            "eye_color": eye_color,
            "hair_color": hair_color,
            "face_shape": face_shape,
            "skin_tone": skin_tone
        }

        logger.info("Traits predicted: %s", traits)
        return jsonify(traits)

    except Exception as e:
        logger.exception("Error processing image")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run()