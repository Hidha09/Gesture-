import tkinter as tk
from tkinter import ttk
import vlc
import os
import time
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
pyautogui.FAILSAFE = False
from tensorflow.keras.models import load_model

# ================================================================
#  CONFIGURATION
# ================================================================
CONFIDENCE_THRESHOLD = 0.88   # minimum confidence to accept a gesture
ACTION_DELAY         = 1.5    # seconds between repeated actions
GESTURE_HOLD_TIME    = 0.6    # seconds hand must be held before action fires
VIDEO_FOLDER         = "videos"

# ================================================================
#  COLORS  (dark theme)
# ================================================================
BG        = "#1a1a2e"
PANEL     = "#16213e"
ACCENT    = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT      = "#eaeaea"
SUBTEXT   = "#a0a0b0"
GREEN     = "#2ecc71"
YELLOW    = "#f39c12"
RED       = "#e74c3c"

# ================================================================
#  MODE
# ================================================================
# "MEDIA" = controls the VLC player
# "SYSTEM" = controls the laptop (YouTube, any app, etc.)
current_mode = "MEDIA"

# ================================================================
#  GESTURE → ACTION MAPS  (one per mode)
# ================================================================
GESTURE_GUIDE_MEDIA = {
    "OPEN_PALM":     ("Play",            GREEN),
    "FIST":          ("Pause",           RED),
    "THUMBS_UP":     ("Next Video",      YELLOW),
    "INDEX_UP":      ("Volume Up",       GREEN),
    "V_SIGN":        ("Volume Down",     YELLOW),
    "OK_SIGN":       ("Enter Fullscreen",TEXT),
    "THREE_FINGERS": ("Exit Fullscreen", RED),
    "POINT_FORWARD": (">> System Mode",  HIGHLIGHT),
}

GESTURE_GUIDE_SYSTEM = {
    "OPEN_PALM":     ("Space (Play/Pause)", GREEN),
    "FIST":          ("Mute",              RED),
    "THUMBS_UP":     ("Next Track",        YELLOW),
    "INDEX_UP":      ("Volume Up",         GREEN),
    "V_SIGN":        ("Volume Down",       YELLOW),
    "OK_SIGN":       ("Enter",             TEXT),
    "THREE_FINGERS": ("Scroll Down",       YELLOW),
    "POINT_FORWARD": ("<< Media Mode",     HIGHLIGHT),
}

# ================================================================
#  LOAD MODEL
# ================================================================
model = load_model("gesture_model.h5")

labels = [
    "FIST", "INDEX_UP", "OK_SIGN", "OPEN_PALM",
    "POINT_FORWARD", "THREE_FINGERS", "THUMBS_UP", "V_SIGN"
]

# ================================================================
#  LOAD VIDEOS
# ================================================================
videos = []
if os.path.exists(VIDEO_FOLDER):
    for file in sorted(os.listdir(VIDEO_FOLDER)):
        if file.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
            videos.append(os.path.join(VIDEO_FOLDER, file))

if not videos:
    # Graceful fallback — show warning, don't crash
    print("WARNING: No videos found in 'videos/' folder.")

current_video = 0

# ================================================================
#  STATE
# ================================================================
last_action_time   = 0
gesture_hold_start = None
last_held_gesture  = None

# ================================================================
#  WINDOW
# ================================================================
root = tk.Tk()
root.title("Gesture Controlled Media Player")
root.geometry("1100x720")
root.configure(bg=BG)
root.resizable(True, True)

# ---- Title Bar ----
title_bar = tk.Frame(root, bg=HIGHLIGHT, height=5)
title_bar.pack(fill="x")

header = tk.Frame(root, bg=BG, pady=8)
header.pack(fill="x", padx=20)

tk.Label(
    header,
    text="Gesture Controlled Media Player",
    font=("Helvetica", 18, "bold"),
    fg=TEXT, bg=BG
).pack(side="left")

# ---- Mode Banner ----
mode_banner = tk.Frame(root, bg=GREEN, pady=4)
mode_banner.pack(fill="x")

mode_banner_label = tk.Label(
    mode_banner,
    text="MODE: MEDIA PLAYER  |  Do POINT FORWARD to switch to System Control",
    font=("Helvetica", 10, "bold"),
    fg=BG, bg=GREEN
)
mode_banner_label.pack()

# ---- Main Layout ----
main = tk.Frame(root, bg=BG)
main.pack(fill="both", expand=True, padx=15, pady=5)

# Left column — video + controls
left = tk.Frame(main, bg=BG)
left.pack(side="left", fill="both", expand=True)

# Right column — scrollable gesture panel
right_outer = tk.Frame(main, bg=PANEL, width=320)
right_outer.pack(side="right", fill="y", padx=(10, 0))
right_outer.pack_propagate(False)

right_canvas = tk.Canvas(right_outer, bg=PANEL, width=300, highlightthickness=0)
right_scroll  = tk.Scrollbar(right_outer, orient="vertical", command=right_canvas.yview)
right_canvas.configure(yscrollcommand=right_scroll.set)

right_scroll.pack(side="right", fill="y")
right_canvas.pack(side="left", fill="both", expand=True)

right = tk.Frame(right_canvas, bg=PANEL, padx=10, pady=10)
right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")

def on_right_configure(e):
    right_canvas.configure(scrollregion=right_canvas.bbox("all"))

right.bind("<Configure>", on_right_configure)

# ================================================================
#  VIDEO AREA
# ================================================================
video_frame = tk.Frame(left, bg="black", width=680, height=400)
video_frame.pack(fill="both", expand=True)
video_frame.pack_propagate(False)

root.update()

# ---- Video name label ----
video_name_var = tk.StringVar(value="No video loaded")
video_name_label = tk.Label(
    left,
    textvariable=video_name_var,
    font=("Helvetica", 10),
    fg=SUBTEXT, bg=BG,
    anchor="w"
)
video_name_label.pack(fill="x", padx=5, pady=(3, 0))

# ================================================================
#  VLC PLAYER
# ================================================================
instance = vlc.Instance()
player   = instance.media_player_new()


def get_video_name(index):
    if not videos:
        return "No video"
    return os.path.basename(videos[index])


def load_video(index):
    if not videos:
        return
    media = instance.media_new(videos[index])
    player.set_media(media)
    # Platform-safe embed
    wid = video_frame.winfo_id()
    if hasattr(player, "set_hwnd"):        # Windows
        player.set_hwnd(wid)
    elif hasattr(player, "set_xwindow"):   # Linux
        player.set_xwindow(wid)
    elif hasattr(player, "set_nsobject"):  # macOS
        player.set_nsobject(wid)
    video_name_var.set(f"▶  {get_video_name(index)}")


def check_video_end():
    if videos and player.get_state() == vlc.State.Ended:
        player.stop()
        load_video(current_video)
        player.play()
    root.after(500, check_video_end)


def play_video():
    if not videos:
        return
    if player.get_state() == vlc.State.Ended:
        load_video(current_video)
    player.play()
    update_volume_display()


def pause_video():
    if not videos:
        return
    if player.get_state() == vlc.State.Ended:
        load_video(current_video)
        player.play()
    else:
        player.pause()


def next_video():
    global current_video
    if not videos:
        return
    current_video = (current_video + 1) % len(videos)
    load_video(current_video)
    player.play()
    update_volume_display()


def prev_video():
    global current_video
    if not videos:
        return
    current_video = (current_video - 1) % len(videos)
    load_video(current_video)
    player.play()


def volume_up():
    vol = player.audio_get_volume()
    new_vol = min(vol + 10, 100)
    player.audio_set_volume(new_vol)
    update_volume_display()


def volume_down():
    vol = player.audio_get_volume()
    new_vol = max(vol - 10, 0)
    player.audio_set_volume(new_vol)
    update_volume_display()


def enter_fullscreen():
    root.attributes("-fullscreen", True)

def toggle_fullscreen():
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))

def exit_fullscreen():
    root.attributes("-fullscreen", False)   # exit tkinter fullscreen
    root.state("normal")                    # also restore from manual maximize


# ================================================================
#  CONTROL BUTTONS
# ================================================================
btn_row = tk.Frame(left, bg=BG, pady=8)
btn_row.pack(fill="x")

BTN_STYLE = dict(
    font=("Helvetica", 11, "bold"),
    fg=TEXT,
    bg=ACCENT,
    activebackground=HIGHLIGHT,
    activeforeground=TEXT,
    relief="flat",
    bd=0,
    padx=14,
    pady=6,
    cursor="hand2"
)

tk.Button(btn_row, text="|<< Prev",    command=prev_video,        **BTN_STYLE).pack(side="left", padx=4)
tk.Button(btn_row, text="Play",        command=play_video,        **BTN_STYLE).pack(side="left", padx=4)
tk.Button(btn_row, text="Pause",       command=pause_video,       **BTN_STYLE).pack(side="left", padx=4)
tk.Button(btn_row, text="Next >>|",    command=next_video,        **BTN_STYLE).pack(side="left", padx=4)
tk.Button(btn_row, text="Vol +",       command=volume_up,         **BTN_STYLE).pack(side="left", padx=4)
tk.Button(btn_row, text="Vol -",       command=volume_down,       **BTN_STYLE).pack(side="left", padx=4)
tk.Button(btn_row, text="Fullscreen",  command=toggle_fullscreen, **BTN_STYLE).pack(side="left", padx=4)

# ---- Volume Bar ----
vol_row = tk.Frame(left, bg=BG)
vol_row.pack(fill="x", padx=5, pady=(0, 5))

tk.Label(vol_row, text="Volume:", fg=SUBTEXT, bg=BG, font=("Helvetica", 9)).pack(side="left")

vol_bar_bg = tk.Frame(vol_row, bg=ACCENT, height=8, width=200)
vol_bar_bg.pack(side="left", padx=8)
vol_bar_bg.pack_propagate(False)

vol_bar_fill = tk.Frame(vol_bar_bg, bg=GREEN, height=8, width=100)
vol_bar_fill.place(x=0, y=0, relheight=1)

vol_label = tk.Label(vol_row, text="50%", fg=TEXT, bg=BG, font=("Helvetica", 9))
vol_label.pack(side="left")


def update_volume_display():
    try:
        vol = player.audio_get_volume()
        if vol < 0:   # VLC returns -1 if player not ready
            vol = 50
        vol_label.config(text=f"{vol}%")
        fill_w = int(200 * vol / 100)
        vol_bar_fill.place(x=0, y=0, relheight=1, width=fill_w)
        if vol > 60:
            vol_bar_fill.config(bg=GREEN)
        elif vol > 30:
            vol_bar_fill.config(bg=YELLOW)
        else:
            vol_bar_fill.config(bg=RED)
    except Exception:
        pass


# ================================================================
#  RIGHT PANEL — GESTURE STATUS + GUIDE
# ================================================================
tk.Label(
    right, text="GESTURE STATUS",
    font=("Helvetica", 10, "bold"),
    fg=HIGHLIGHT, bg=PANEL
).pack(anchor="w", pady=(0, 6))

# Current gesture display
gesture_box = tk.Frame(right, bg=ACCENT, pady=10, padx=10)
gesture_box.pack(fill="x", pady=(0, 10))

gesture_name_var = tk.StringVar(value="None")
gesture_name_label = tk.Label(
    gesture_box,
    textvariable=gesture_name_var,
    font=("Helvetica", 14, "bold"),
    fg=TEXT, bg=ACCENT
)
gesture_name_label.pack()

action_var = tk.StringVar(value="—")
action_label = tk.Label(
    gesture_box,
    textvariable=action_var,
    font=("Helvetica", 10),
    fg=YELLOW, bg=ACCENT
)
action_label.pack()

# Confidence bar
tk.Label(right, text="Confidence:", fg=SUBTEXT, bg=PANEL, font=("Helvetica", 8)).pack(anchor="w")

conf_bar_bg = tk.Frame(right, bg=ACCENT, height=8)
conf_bar_bg.pack(fill="x", pady=(2, 8))
conf_bar_bg.pack_propagate(False)

conf_bar_fill = tk.Frame(conf_bar_bg, bg=GREEN, height=8, width=0)
conf_bar_fill.place(x=0, y=0, relheight=1, width=0)

conf_label = tk.Label(right, text="0%", fg=SUBTEXT, bg=PANEL, font=("Helvetica", 8))
conf_label.pack(anchor="w")

# Hold progress bar
tk.Label(right, text="Hold progress:", fg=SUBTEXT, bg=PANEL, font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))

hold_bar_bg = tk.Frame(right, bg=ACCENT, height=8)
hold_bar_bg.pack(fill="x", pady=(2, 12))
hold_bar_bg.pack_propagate(False)

hold_bar_fill = tk.Frame(hold_bar_bg, bg=YELLOW, height=8, width=0)
hold_bar_fill.place(x=0, y=0, relheight=1, width=0)

# Separator
tk.Frame(right, bg=ACCENT, height=1).pack(fill="x", pady=6)

# Gesture Guide — dynamically updated on mode switch
guide_title = tk.Label(
    right, text="GESTURE GUIDE",
    font=("Helvetica", 10, "bold"),
    fg=HIGHLIGHT, bg=PANEL
)
guide_title.pack(anchor="w", pady=(4, 6))

guide_frame = tk.Frame(right, bg=PANEL)
guide_frame.pack(fill="both", expand=True)

def refresh_guide():
    """Rebuild the gesture guide panel for the current mode."""
    for widget in guide_frame.winfo_children():
        widget.destroy()
    guide = GESTURE_GUIDE_MEDIA if current_mode == "MEDIA" else GESTURE_GUIDE_SYSTEM
    for i, (gesture, (desc, color)) in enumerate(guide.items()):
        tk.Label(guide_frame, text=gesture,
                 font=("Courier", 9, "bold"), fg=color, bg=PANEL,
                 anchor="w").grid(row=i, column=0, sticky="w", pady=2, padx=(0, 8))
        tk.Label(guide_frame, text=desc,
                 font=("Helvetica", 9), fg=TEXT, bg=PANEL,
                 anchor="w").grid(row=i, column=1, sticky="w", pady=2)
    guide_frame.columnconfigure(1, weight=1)

refresh_guide()

# Status indicator at bottom
tk.Frame(right, bg=ACCENT, height=1).pack(fill="x", pady=8)

cam_status_var = tk.StringVar(value="Camera Active")
tk.Label(right, textvariable=cam_status_var, font=("Helvetica", 9), fg=GREEN, bg=PANEL).pack(anchor="w")

# ================================================================
#  MEDIAPIPE + CAMERA SETUP
# ================================================================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cam_status_var.set("Camera Not Found")


# ================================================================
#  LANDMARK-BASED GESTURE CORRECTION
# ================================================================
def is_finger_extended(hand_landmarks, tip_id, margin=0.04):
    """Returns True if finger tip is clearly above its middle joint."""
    return (hand_landmarks.landmark[tip_id].y
            < hand_landmarks.landmark[tip_id - 2].y - margin)

def correct_gesture(raw_gesture, hand_landmarks, hand_label="Left"):
    lm = hand_landmarks.landmark

    # ---- Never override OK_SIGN ----
    if raw_gesture == "OK_SIGN":
        return "OK_SIGN"

    index_ext  = is_finger_extended(hand_landmarks, 8)
    middle_ext = is_finger_extended(hand_landmarks, 12)
    ring_ext   = is_finger_extended(hand_landmarks, 16)
    pinky_ext  = is_finger_extended(hand_landmarks, 20)

    # Thumb direction depends on which hand
    # MediaPipe labels are from the CAMERA's perspective (mirrored)
    # Left hand in mirror = user's Right hand → thumb tip is to the RIGHT (higher x)
    if hand_label == "Left":
        # user's right hand (mirrored) — thumb tip x > base x means open
        thumb_open = lm[4].x > lm[3].x + 0.02
    else:
        # user's left hand (mirrored) — thumb tip x < base x means open
        thumb_open = lm[4].x < lm[3].x - 0.02

    extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    # ---- OPEN_PALM vs THREE_FINGERS ----
    if raw_gesture in ("OPEN_PALM", "THREE_FINGERS"):
        if extended_count == 4 and thumb_open:
            return "OPEN_PALM"
        elif extended_count <= 3 and not thumb_open:
            return "THREE_FINGERS"

    # ---- INDEX_UP vs POINT_FORWARD ----
    if raw_gesture == "INDEX_UP":
        tip_y  = lm[8].y
        base_y = lm[5].y
        if abs(tip_y - base_y) < 0.05:
            return "POINT_FORWARD"

    return raw_gesture


# ================================================================
#  MODE TOGGLE
# ================================================================
def toggle_mode():
    global current_mode
    if current_mode == "MEDIA":
        current_mode = "SYSTEM"
        mode_banner.config(bg=HIGHLIGHT)
        mode_banner_label.config(
            text="MODE: SYSTEM CONTROL  |  Do POINT FORWARD to switch back to Media Player",
            bg=HIGHLIGHT, fg=TEXT
        )
    else:
        current_mode = "MEDIA"
        mode_banner.config(bg=GREEN)
        mode_banner_label.config(
            text="MODE: MEDIA PLAYER  |  Do POINT FORWARD to switch to System Control",
            bg=GREEN, fg=BG
        )
    refresh_guide()
    print(f"Switched to {current_mode} mode")


# ================================================================
#  GESTURE DETECTION LOOP
# ================================================================
def detect_gesture():
    global last_action_time, gesture_hold_start, last_held_gesture

    try:
        ret, frame = cap.read()
        if not ret:
            cam_status_var.set("Camera Error")
            root.after(100, detect_gesture)
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb_frame)

        current_gesture = "None"
        confidence      = 0.0

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

                # Get hand label: "Left" or "Right" (camera-perspective)
                hand_label = "Left"
                if results.multi_handedness:
                    hand_label = results.multi_handedness[i].classification[0].label

                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                landmark_array  = np.array(landmark_list).reshape(1, -1)
                prediction      = model.predict(landmark_array, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence      = float(np.max(prediction))

                if confidence > CONFIDENCE_THRESHOLD:
                    raw_gesture     = labels[predicted_class]
                    current_gesture = correct_gesture(raw_gesture, hand_landmarks, hand_label)

        # ---- Update UI ----
        gesture_name_var.set(current_gesture)

        conf_pct = int(confidence * 100)
        conf_label.config(text=f"{conf_pct}%")
        conf_w = int(200 * confidence)
        conf_bar_fill.place(x=0, y=0, relheight=1, width=conf_w)
        conf_bar_fill.config(bg=GREEN if confidence > 0.9 else YELLOW if confidence > 0.7 else RED)

        # Pick the right guide based on current mode
        active_guide = GESTURE_GUIDE_MEDIA if current_mode == "MEDIA" else GESTURE_GUIDE_SYSTEM
        if current_gesture in active_guide:
            desc, color = active_guide[current_gesture]
            action_var.set(desc)
            action_label.config(fg=color)
            gesture_name_label.config(fg=color)
        else:
            action_var.set("—")
            action_label.config(fg=SUBTEXT)
            gesture_name_label.config(fg=TEXT)

        # ---- Hold-to-confirm system ----
        now = time.time()

        if current_gesture != "None":
            if current_gesture != last_held_gesture:
                gesture_hold_start = now
                last_held_gesture  = current_gesture

            hold_elapsed = now - gesture_hold_start
            hold_pct     = min(hold_elapsed / GESTURE_HOLD_TIME, 1.0)
            hold_w       = int(200 * hold_pct)
            hold_bar_fill.place(x=0, y=0, relheight=1, width=hold_w)
            hold_bar_fill.config(bg=GREEN if hold_pct >= 1.0 else YELLOW)

            if hold_elapsed >= GESTURE_HOLD_TIME and (now - last_action_time) >= ACTION_DELAY:

                print(f"[{current_mode}] ACTION: {current_gesture} ({confidence:.2f})")

                # ---- POINT_FORWARD: always toggles mode ----
                if current_gesture == "POINT_FORWARD":
                    toggle_mode()

                # ---- MEDIA mode actions ----
                elif current_mode == "MEDIA":
                    if current_gesture == "OPEN_PALM":
                        play_video()
                    elif current_gesture == "FIST":
                        pause_video()
                    elif current_gesture == "THUMBS_UP":
                        next_video()
                    elif current_gesture == "INDEX_UP":
                        volume_up()
                    elif current_gesture == "V_SIGN":
                        volume_down()
                    elif current_gesture == "OK_SIGN":
                        enter_fullscreen()
                    elif current_gesture == "THREE_FINGERS":
                        exit_fullscreen()

                # ---- SYSTEM mode actions ----
                elif current_mode == "SYSTEM":
                    if current_gesture == "OPEN_PALM":
                        pyautogui.press("space")
                    elif current_gesture == "FIST":
                        pyautogui.press("volumemute")
                    elif current_gesture == "THUMBS_UP":
                        pyautogui.press("nexttrack")
                    elif current_gesture == "INDEX_UP":
                        pyautogui.press("volumeup")
                    elif current_gesture == "V_SIGN":
                        pyautogui.press("volumedown")
                    elif current_gesture == "OK_SIGN":
                        pyautogui.press("enter")
                    elif current_gesture == "THREE_FINGERS":
                        pyautogui.scroll(-300)

                last_action_time   = now
                gesture_hold_start = now

        else:
            gesture_hold_start = None
            last_held_gesture  = None
            hold_bar_fill.place(x=0, y=0, relheight=1, width=0)

    except Exception as e:
        print(f"[detect_gesture error] {e}")

    root.after(50, detect_gesture)


# ================================================================
#  KEYBOARD SHORTCUTS (Escape to exit fullscreen)
# ================================================================
root.bind("<Escape>", lambda e: exit_fullscreen())
root.bind("<space>",  lambda e: pause_video())
root.bind("<Right>",  lambda e: next_video())
root.bind("<Left>",   lambda e: prev_video())
root.bind("<Up>",     lambda e: volume_up())
root.bind("<Down>",   lambda e: volume_down())

# ================================================================
#  START
# ================================================================
if videos:
    load_video(current_video)

update_volume_display()
detect_gesture()
check_video_end()

root.mainloop()

cap.release()
cv2.destroyAllWindows()