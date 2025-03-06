from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("runs/detect/train9/weights/best.pt")
camera = cv2.VideoCapture(0)

# Get class names from model
class_names = model.names  # Dictionary of class IDs and names

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run YOLO model
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])  
                label = class_names[class_id]  
                conf = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Ensure label is drawn properly
                text = f"{label} ({conf:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20  # Prevent text from going off-frame

                # Draw filled rectangle for text background
                cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), cv2.FILLED)

                # Put label text on bounding box
                cv2.putText(frame, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_live')
def predict_live():
    success, frame = camera.read()
    if not success:
        return jsonify({"result": "No frame detected"})

    results = model(frame)
    detected_classes = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = class_names[class_id]
            detected_classes.append(label)

    return jsonify({"result": ", ".join(detected_classes) if detected_classes else "No waste detected"})

if __name__ == '__main__':
    app.run(debug=True)
