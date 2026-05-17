import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time

gesture_start_time = None
confirmation_time = 1.0   # seconds

# ---------------- Load Model ----------------
model = load_model("gesture_model.h5")

labels = [
    "FIST",
    "INDEX_UP",
    "OK_SIGN",
    "OPEN_PALM",
    "POINT_FORWARD",
    "THREE_FINGERS",
    "THUMBS_UP",
    "V_SIGN"
]

encoder = LabelEncoder()
encoder.fit(labels)

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False

confidence_threshold = 0.92

# ---------------- Main Loop ----------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = ""

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # -------- Landmark Processing --------
            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            landmark_array = np.array(landmark_list).reshape(1, -1)

            prediction = model.predict(landmark_array, verbose=0)

            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > confidence_threshold:

                current_gesture = encoder.inverse_transform([predicted_class])[0]

                # -------- OPEN PALM vs THREE FINGERS FIX --------
                finger_tips = [8, 12, 16, 20]
                extended_fingers = 0

                for tip in finger_tips:
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                        extended_fingers += 1

                thumb_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x

                if extended_fingers == 4 and thumb_open:
                    current_gesture = "OPEN_PALM"

                # -------- INDEX UP vs POINT FORWARD FIX --------
                if current_gesture == "INDEX_UP":

                    index_tip_y = hand_landmarks.landmark[8].y
                    index_base_y = hand_landmarks.landmark[5].y

                    if abs(index_tip_y - index_base_y) < 0.05:
                        current_gesture = "POINT_FORWARD"

                cv2.putText(frame,
            f"Gesture: {current_gesture} ({confidence:.2f})",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)

    # -------- Gesture Confirmation System --------
    if current_gesture != "":

        if gesture_start_time is None:
            gesture_start_time = time.time()

        hold_time = time.time() - gesture_start_time

        cv2.putText(
            frame,
            f"Holding: {hold_time:.1f}s",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if hold_time > confirmation_time:

            print("CONFIRMED:", current_gesture)

            if current_gesture == "OPEN_PALM":
                pyautogui.press("space")

            elif current_gesture == "FIST":
                pyautogui.press("esc")

            elif current_gesture == "INDEX_UP":
                pyautogui.press("volumeup")

            elif current_gesture == "V_SIGN":
                pyautogui.press("volumedown")

            elif current_gesture == "THUMBS_UP":
                pyautogui.press("right")

            elif current_gesture == "POINT_FORWARD":
                pyautogui.press("tab")

            elif current_gesture == "THREE_FINGERS":
                pyautogui.scroll(-300)

            elif current_gesture == "OK_SIGN":
                pyautogui.press("enter")

            gesture_start_time = None

    else:
        gesture_start_time = None

    # -------- Interface Title --------
    cv2.putText(
        frame,
        "Intelligent Gesture-Controlled Virtual Interface",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # -------- Instructions Panel --------
   

    cv2.imshow("AI Gesture Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()