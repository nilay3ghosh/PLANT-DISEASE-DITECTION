import cv2
import os
from ultralytics import YOLO

# -------------------------------
# Paths (USE ABSOLUTE PATHS)
# -------------------------------
MODEL_PATH = r"D:\Final Year Proj\Plant Disease Detection\Models\Model1\weights\best.pt"
VIDEO_PATH = r"D:\Final Year Proj\Plant Disease Detection\Samples\Potato Disease02.mp4"

# -------------------------------
# Load Model
# -------------------------------
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully")

# -------------------------------
# Check Video Path
# -------------------------------
if not os.path.exists(VIDEO_PATH):
    print("❌ Video path incorrect")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

print("✅ Video opened successfully")

# -------------------------------
# Video Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ End of video")
        break

    # YOLO inference
    results = model(frame, conf=0.25, verbose=False)

    # Draw results
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Video Test", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
