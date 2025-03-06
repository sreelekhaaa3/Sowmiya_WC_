import torch
from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("runs/detect/train9/weights/best.pt")

def classify_image(image_path):
    frame = cv2.imread(image_path)
    results = model(frame)

    detected_classes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            detected_classes.append(label)

    return detected_classes

if __name__ == "__main__":
    image_path = r"M:\DHAMU\Sowmiya\dataset\images\val\cardboard1_jpg.rf.c238a492c67375958c9320cff493dde6.jpg"  # Use raw string (r"")
    classes = classify_image(image_path)
    print(f"Detected: {', '.join(classes)}" if classes else "No waste detected.")

