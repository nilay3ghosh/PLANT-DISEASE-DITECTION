# -------------------------------
# Import Libraries
# -------------------------------
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# -------------------------------
# Load Models
# -------------------------------
yolo_model = YOLO("Model/PotatoDisease/best.pt")
classifier = load_model("Model/EfficientNetV2_PotatoDisease.h5")

# -------------------------------
# Class Labels
# -------------------------------
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy", "Other"]
IMG_SIZE = (224, 224)

# -------------------------------
# EfficientNetV2 Classification
# -------------------------------
def classify_crop(crop):
    crop = cv2.resize(crop, IMG_SIZE)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = np.expand_dims(crop, axis=0)
    crop = preprocess_input(crop)

    preds = classifier.predict(crop, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])
    return CLASS_NAMES[class_id], confidence

# -------------------------------
# YOLO + EfficientNet Video Processing
# -------------------------------
def process_frame(frame, det_conf=0.25, cls_conf=0.60):
    results = yolo_model(frame, conf=det_conf, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            disease, conf = classify_crop(crop)

            if conf < cls_conf:
                continue

            label = f"{disease} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    return frame

# -------------------------------
# Main Video Loop
# -------------------------------
def run_video(video_path=0):
    """
    video_path: int for webcam, or str path to video file
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = process_frame(frame)

        cv2.imshow("YOLOv8 + EfficientNetV2 Video", output_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    # For webcam use: run_video(0)
    # For video file use: run_video("test_video.mp4")
    run_video("test_video.mp4")
