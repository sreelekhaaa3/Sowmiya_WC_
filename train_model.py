from ultralytics import YOLO

# Load YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. based on your preference)
model = YOLO("yolov8n.pt")  # Using the nano model for faster training

# Train the model
model.train(
    data="M:/DHAMU/Sowmiya/waste_dataset.yaml",  # Path to your dataset configuration file
    epochs=3,   # Adjust as needed
    imgsz=640,   # Image size
    batch=4,     # Adjust based on your GPU
    device="cpu"  # Use "cpu" if you don't have a GPU
)
