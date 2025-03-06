import cv2

def capture_frame():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite("captured.jpg", frame)  # Save the image
        return "captured.jpg"
    else:
        print("Error: Could not capture frame.")
        return None

if __name__ == "__main__":
    capture_frame()
