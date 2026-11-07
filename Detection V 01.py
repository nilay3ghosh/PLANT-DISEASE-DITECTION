import cv2
from ultralytics import YOLO

# -------------------------------
# Load YOLOv8 Model
# -------------------------------
yolo_model = YOLO("Plant Disease Detection/Models/Model1/weights/best.pt")

# -------------------------------
# Class Names
# -------------------------------
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy", "Other"]

# -------------------------------
# Video Detection Function
# -------------------------------
def run_video(video_path=0, conf_threshold=0.25):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = yolo_model(frame, conf=conf_threshold, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                label = f"{CLASS_NAMES[int(cls_id)]} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10 if y1 > 20 else y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("YOLOv8 Detection Only", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    # Webcam
    # run_video(0)

    # Video file
    run_video("Plant Disease Detection/Samples/Potato Disease01.mp4")
