import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_palm_side(lm, hand_label):
    """
    Estimate if palm or back is facing the camera.
    
    Args:
        lm: List of landmarks for the hand.
        hand_label: "Left" or "Right" (from MediaPipe results).
        
    Returns:
        "palm" or "back"
    """
    wrist = lm[0]
    index_mcp = lm[5]
    pinky_mcp = lm[17]

    # 1. Create vectors in 3D: (Index - Wrist) and (Pinky - Wrist)
    # This creates a plane defined by the palm.
    v1 = (
        index_mcp.x - wrist.x,
        index_mcp.y - wrist.y,
        index_mcp.z - wrist.z
    )
    v2 = (
        pinky_mcp.x - wrist.x,
        pinky_mcp.y - wrist.y,
        pinky_mcp.z - wrist.z
    )

    # 2. Calculate Cross product (v1 x v2) -> Normal Vector (nz)
    # We only care about the Z-component (depth direction)
    nz = v1[0] * v2[1] - v1[1] * v2[0]

    # 3. Determine side based on hand label
    # The geometric order of Index vs Pinky is swapped for Left vs Right hands.
    if hand_label == "Left":
        # For Left hand: Negative Z usually indicates Palm facing camera
        return "palm" if nz < 0 else "back"
    else:
        # For Right hand: Positive Z usually indicates Palm facing camera
        return "palm" if nz > 0 else "back"


# --- Main Application ---
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror view (makes interaction more natural)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Process frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                # Get the label ("Left" or "Right")
                # Note: In mirror mode, MediaPipe might label your physical Right hand as "Left" 
                # depending on how it perceives the flip. We trust MP's label for the math.
                hand_label = results.multi_handedness[idx].classification[0].label
                lm = hand_landmarks.landmark

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # 1. Detect Side (Palm vs Back)
                side = get_palm_side(lm, hand_label)

                fingers_up = []

                # 2. Thumb Logic (Complex because thumb moves sideways)
                # Tip=4, Joint=3.
                thumb_tip = lm[4]
                thumb_joint = lm[3]
                is_thumb_up = False

                # Logic table based on Hand Label + Side + Mirror View
                if side == "palm":
                    if hand_label == "Right":
                        # Right Palm: Thumb is on the LEFT side of the hand (smaller X)
                        is_thumb_up = thumb_tip.x < thumb_joint.x
                    else: # Left Palm
                        # Left Palm: Thumb is on the RIGHT side of the hand (larger X)
                        is_thumb_up = thumb_tip.x > thumb_joint.x
                else: # Back
                    if hand_label == "Right":
                        # Right Back: Thumb is on the RIGHT side of the hand
                        is_thumb_up = thumb_tip.x > thumb_joint.x
                    else: # Left Back
                        # Left Back: Thumb is on the LEFT side of the hand
                        is_thumb_up = thumb_tip.x < thumb_joint.x
                
                fingers_up.append(is_thumb_up)

                # 3. Other 4 Fingers Logic (Simple vertical check)
                # Tips: 8(Index), 12(Middle), 16(Ring), 20(Pinky)
                # PIPs: 6, 10, 14, 18 (Lower joints)
                finger_tips = [8, 12, 16, 20]
                finger_pips = [6, 10, 14, 18]

                for tip_id, pip_id in zip(finger_tips, finger_pips):
                    # In image (0,0) is top-left. So Y gets smaller as you go UP.
                    # If Tip Y < Pip Y, the finger is raised.
                    fingers_up.append(lm[tip_id].y < lm[pip_id].y)

                count = sum(fingers_up)

                # 4. Display Info
                wrist_x = int(lm[0].x * w)
                wrist_y = int(lm[0].y * h)

                # Choose color: Green for Palm, Red for Back
                text_color = (0, 255, 0) if side == "palm" else (0, 0, 255)

                cv2.putText(
                    frame,
                    f"{hand_label} {side}: {count}",
                    (wrist_x - 60, wrist_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2
                )

        cv2.imshow("Finger Counter (Palm/Back Corrected)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
