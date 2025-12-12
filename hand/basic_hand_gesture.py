import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. SETUP ---
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Define your custom names here
custom_labels = {
    "None": "Waiting...",
    "Closed_Fist": "Rock",
    "Open_Palm": "Paper",
    "Pointing_Up": "Look Up",
    "Thumb_Down": "Bad",
    "Thumb_Up": "Good",
    "Victory": "Scissors",
    "ILoveYou": "Rock n Roll"
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror view
    frame = cv2.flip(frame,1)
    
    # Prepare image for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # --- 2. RECOGNIZE ---
    result = recognizer.recognize(mp_image)

    # --- 3. CUSTOMIZE & DISPLAY ---
    if result.gestures:
        # Get the first gesture detected
        mp_category = result.gestures[0][0].category_name
        confidence = result.gestures[0][0].score

        # Translate to your custom name
        display_text = custom_labels.get(mp_category, mp_category)

        # Draw on screen
        cv2.putText(frame, 
                    f"Gesture: {display_text}", 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        cv2.putText(frame, 
                    f"Confidence: {int(confidence * 100)}%", 
                    (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 1)

    cv2.imshow('Custom Gesture Names', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
