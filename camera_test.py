import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

previous_gesture = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            fingers = [index_up, middle_up, ring_up, pinky_up]
            total_fingers = fingers.count(True)

            if total_fingers == 4:
                current_gesture = "OPEN PALM"

            elif total_fingers == 0:
                current_gesture = "FIST"

            elif index_up and total_fingers == 1:
                current_gesture = "ONLY INDEX UP"

            if current_gesture != "":
                cv2.putText(frame, current_gesture, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 3)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Trigger action only when gesture changes
    if current_gesture != previous_gesture:

        if current_gesture == "OPEN PALM":
            pyautogui.press("space")

        elif current_gesture == "FIST":
            pyautogui.press("esc")

        previous_gesture = current_gesture

    cv2.imshow("Gesture Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
