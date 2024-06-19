import cv2
import torch
import math

# Load the pre-trained Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# To use local haarcascade_eye.xml file
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

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
                face_center_y = int(y1) + (int(y2) - int(y1)) // 2

                # Calculate the angle between the eye center and face center
                dx = eye_center[0] - face_center_x
                dy = eye_center[1] - face_center_y
                angle = math.atan2(dy, dx) * 180 / math.pi

                # Determine the direction based on the angle
                if -22.5 <= angle < 22.5:
                    direction = "Right"
                elif 22.5 <= angle < 67.5:
                    direction = "Up-Right"
                elif 67.5 <= angle <= 112.5:
                    direction = "Up"
                elif 112.5 < angle <= 157.5:
                    direction = "Up-Left"
                elif abs(angle) > 157.5:
                    direction = "Left"
                elif -157.5 <= angle < -112.5:
                    direction = "Down-Left"
                elif -112.5 <= angle < -67.5:
                    direction = "Down"
                else:
                    direction = "Down-Right"

                cv2.putText(frame, f"Looking {direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Eye Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
