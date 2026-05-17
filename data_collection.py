import cv2
import mediapipe as mp
import csv
import os
from collections import Counter

# ================================================================
#  CONFIGURATION
# ================================================================
FILE_NAME = "gesture_data.csv"

# Only collecting for the 4 under-represented gestures
GESTURES = ["OK_SIGN", "THREE_FINGERS", "V_SIGN", "POINT_FORWARD"]
TARGET_PER_GESTURE = 200   # we want 200 total for each gesture

# ================================================================
#  MEDIAPIPE SETUP
# ================================================================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ================================================================
#  LOAD EXISTING COUNTS
# ================================================================
def get_existing_counts():
    counts = Counter()
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    counts[row[-1]] += 1
    return counts

# ================================================================
#  STATE
# ================================================================
current_index  = 0
sample_count   = 0
existing       = get_existing_counts()

# ================================================================
#  MAIN LOOP
# ================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_label   = GESTURES[current_index]
    already_have    = existing[current_label]
    still_need      = max(0, TARGET_PER_GESTURE - already_have)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(rgb_frame)

    hand_detected  = False
    landmark_list  = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            hand_detected = True

    # ---- UI ----
    h, w = frame.shape[:2]

    # Background panel
    cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 40), -1)

    # Current gesture
    cv2.putText(frame, f"Gesture: {current_label}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)

    # Progress
    progress_text = f"Have: {already_have}  |  Need: {still_need} more  |  Target: {TARGET_PER_GESTURE}"
    cv2.putText(frame, progress_text,
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Instructions
    cv2.putText(frame, "S = Save sample   N = Next gesture   Q = Quit",
                (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)

    # Hand status
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text  = "Hand detected" if hand_detected else "No hand detected"
    cv2.putText(frame, status_text,
                (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Progress bar
    if TARGET_PER_GESTURE > 0:
        bar_w     = w - 20
        filled_w  = int(bar_w * min(already_have / TARGET_PER_GESTURE, 1.0))
        cv2.rectangle(frame, (10, 145), (10 + bar_w, 158), (50, 50, 50), -1)
        bar_color = (0, 255, 100) if already_have >= TARGET_PER_GESTURE else (0, 180, 255)
        cv2.rectangle(frame, (10, 145), (10 + filled_w, 158), bar_color, -1)

    # Done indicator
    if still_need == 0:
        cv2.putText(frame, "DONE! Press N for next gesture",
                    (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # All gestures summary on right
    for i, g in enumerate(GESTURES):
        count  = existing[g]
        done   = count >= TARGET_PER_GESTURE
        color  = (0, 255, 100) if done else (0, 180, 255)
        prefix = "[DONE]" if done else f"[{count}/{TARGET_PER_GESTURE}]"
        marker = ">>>" if i == current_index else "   "
        cv2.putText(frame, f"{marker} {prefix} {g}",
                    (w - 280, 40 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # S — save sample
    if key == ord('s') and hand_detected and landmark_list:
        with open(FILE_NAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(landmark_list + [current_label])
        existing[current_label] += 1
        already_have = existing[current_label]
        sample_count += 1
        print(f"Saved {current_label} sample #{already_have}")

    # N — next gesture
    elif key == ord('n'):
        current_index = (current_index + 1) % len(GESTURES)
        print(f"Switched to: {GESTURES[current_index]}")

    # Q — quit
    elif key == ord('q'):
        break

print("\n--- Final counts ---")
for g in GESTURES:
    print(f"{g}: {existing[g]}")

cap.release()
cv2.destroyAllWindows()