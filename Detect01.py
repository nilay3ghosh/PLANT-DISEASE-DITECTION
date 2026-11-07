import cv2
from ultralytics import YOLO

# -------------------------------
# Load YOLOv8 Model
# -------------------------------
model_path = "Plant Disease Detection/Models/Model3/weights/best.pt"
model = YOLO(model_path)

print("✅ Model loaded successfully")

# -------------------------------
# Test Image Path
# -------------------------------
image_path = "Plant Disease Detection/Samples/Potato Leaf02.jpg"

img = cv2.imread(image_path)
if img is None:
    print("❌ Error: Image not found or cannot be read")
    exit()

print("✅ Image loaded successfully")

# -------------------------------
# Run Inference
# -------------------------------
results = model(img, conf=0.25, verbose=False)

# -------------------------------
# Print Results
# -------------------------------
for r in results:
    if r.boxes is None:
        print("⚠️ No detections")
        continue

    for cls_id, conf in zip(r.boxes.cls, r.boxes.conf):
        print(f"Detected: class={int(cls_id)}  confidence={conf:.2f}")

# -------------------------------
# Show Output
# -------------------------------
annotated_img = results[0].plot()
cv2.imshow("YOLOv8 Model Test", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
