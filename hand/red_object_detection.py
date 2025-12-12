import cv2
import numpy as np

# Open default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green color range in HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])



    # Masks for both red ranges
    green_mask = cv2.inRange(hsv, lower_green, upper_green)


    # Find contours on the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around red areas (like faces)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # ignore small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "GREEN", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show live detection
    cv2.imshow("GREEN Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
