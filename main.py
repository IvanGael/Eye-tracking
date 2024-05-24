import cv2
import torch

# Load the pre-trained Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the YOLOv5 model (use a pre-trained model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection with YOLOv5
    results = model(frame)

    # Parse the results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        if cls == 0:  # Class 0 is for person
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            face_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Convert the face ROI to grayscale for Haar cascade
            gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray_face_roi)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eye_center = (int(x1) + ex + ew // 2, int(y1) + ey + eh // 2)
                cv2.circle(frame, eye_center, 2, (0, 0, 255), -1)
                
                face_center_x = int(x1) + (int(x2) - int(x1)) // 2
                if eye_center[0] < face_center_x + 10:
                    cv2.putText(frame, "Looking left", (-50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif eye_center[0] > face_center_x + 10:
                    cv2.putText(frame, "Looking right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Eye Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
