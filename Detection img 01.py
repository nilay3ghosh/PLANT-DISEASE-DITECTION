# -------------------------------
# Import Libraries
# -------------------------------
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# -------------------------------
# Load Trained Models
# -------------------------------

# YOLOv8 trained detector
yolo_model = YOLO("Model/PotatoDisease/best.pt")

# EfficientNetV2 trained classifier (DO NOT overwrite)
classifier = load_model("Model/EfficientNetV2_PotatoDisease.h5")

# -------------------------------
# Class Labels
# -------------------------------
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy",
    "Other"
]

IMG_SIZE = (224, 224)

# -------------------------------
# EfficientNetV2 Classification
# -------------------------------
def classify_crop(crop):
    """
    Classify a cropped disease region using EfficientNetV2
    """
    crop = cv2.resize(crop, IMG_SIZE)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = np.expand_dims(crop, axis=0)
    crop = preprocess_input(crop)

    preds = classifier.predict(crop, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])

    return CLASS_NAMES[class_id], confidence


# -------------------------------
# YOLOv8 Detection + Classification
# -------------------------------
def detect_and_classify(image_path, det_conf=0.25, cls_conf=0.60):
    """
    Detect disease regions using YOLOv8
    and classify them using EfficientNetV2
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid path")

    results = yolo_model(image, conf=det_conf, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Safety check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            disease, conf = classify_crop(crop)

            # Optional confidence threshold
            if conf < cls_conf:
                continue

            # Draw bounding box and label
            label = f"{disease} ({conf:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return image


# -------------------------------
# Run the Hybrid Model
# -------------------------------
if __name__ == "__main__":
    input_image = "test_leaf.jpg"

    output = detect_and_classify(input_image)

    cv2.imshow("YOLOv8 + EfficientNetV2 Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
