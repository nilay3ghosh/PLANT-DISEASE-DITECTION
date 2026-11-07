# -------------------------------
# Import Libraries
# -------------------------------
import cv2
from ultralytics import YOLO

# -------------------------------
# Load YOLOv8 Model
# -------------------------------
# This should be trained on your potato disease dataset
yolo_model = YOLO("Model/PotatoDisease/best.pt")

# -------------------------------
# Class Names (Optional, if trained with multiple disease classes)
# -------------------------------
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy", "Other"]

# -------------------------------
# Detection Function (Image)
# -------------------------------
def detect_diseases(image_path, conf_threshold=0.25):
    """
    Detect disease regions in an image using YOLOv8
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid path")

    # Run YOLO detection
    results = yolo_model(image, conf=conf_threshold, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            label = f"{CLASS_NAMES[int(cls_id)]} ({conf:.2f})"

            # Draw bounding box
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
# Run on Image
# -------------------------------
if __name__ == "__main__":
    output_image = detect_diseases("test_leaf.jpg")
    cv2.imshow("YOLOv8 Detection Only", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
