# 🤖 Intelligent Gesture-Controlled Virtual Interface

A real-time hand gesture recognition system that lets you control your computer using hand gestures captured through a standard webcam — no special hardware required.

Built using **Google MediaPipe**, **TensorFlow/Keras**, **OpenCV**, and **Tkinter**.

---

## 📸 Demo

> Control your media player and system applications using only your hand gestures in real time.

| Gesture | Media Player Mode | System Control Mode |
|---|---|---|
| ✋ Open Palm | Play video | Space (play/pause) |
| ✊ Fist | Pause video | Mute system volume |
| 👍 Thumbs Up | Next video | Next track |
| ☝️ Index Up | Volume up | System volume up |
| ✌️ V Sign | Volume down | System volume down |
| 👌 OK Sign | Enter fullscreen | Enter key |
| 🤟 Three Fingers | Exit fullscreen | Scroll down |
| 👉 Point Forward | Switch to System mode | Switch to Media mode |

---

## ✨ Features

- **8 gesture classes** recognized in real time using a trained Keras MLP model
- **Dual-mode architecture** — Media Player mode (VLC) and System Control mode (any app)
- **Mode switching** via Point Forward gesture — works on YouTube, Spotify, any application
- **Hold-to-confirm** — gesture must be held for 0.6 seconds before action fires
- **88% confidence threshold** — low-confidence frames are discarded automatically
- **Landmark correction layer** — resolves confusion between similar gestures
- **Both hands supported** — handedness-aware logic for left and right hand
- **Professional dark GUI** — real-time gesture status, confidence bar, hold bar, mode banner
- **100% test accuracy** after balanced data collection and retraining
- **No GPU required** — runs entirely on CPU on any standard laptop

---

## 🗂️ Project Structure

```
gesture-controlled-interface/
│
├── gesture_interface.py       # Main application — run this
├── data_collection.py         # Script to collect training data
├── retrain_model.py           # Script to retrain the model
├── gesture_model.h5           # Trained Keras MLP model
├── gesture_model_backup.h5    # Backup of trained model
├── gesture_data.csv           # Training data (landmarks + labels)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── videos/                    # Place your video files here
    ├── video1.mp4
    └── ...
```

---

## ⚙️ How It Works

```
Webcam Frame
     ↓
MediaPipe Hands
(21 landmarks × 3 coords = 63 values)
     ↓
Keras MLP Model
(Softmax over 8 gesture classes)
     ↓
Confidence > 88%?
     ↓
Landmark Correction Layer
(Resolve Open Palm vs Three Fingers, handedness)
     ↓
Hold-to-Confirm (0.6s) + Cooldown (1.5s)
     ↓
┌─────────────────┬──────────────────────┐
│  Media Player   │   System Control     │
│  (VLC command)  │   (PyAutoGUI key)    │
└─────────────────┴──────────────────────┘
     ↓
Tkinter GUI Update
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or 3.10
- VLC Media Player installed on your system
- A working webcam (built-in or USB)
- Windows 10/11 recommended (also works on Linux and macOS)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/gesture-controlled-interface.git
cd gesture-controlled-interface
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your videos

```bash
mkdir videos
# Copy your video files into the videos/ folder
```

### 4. Run the application

```bash
python gesture_interface.py
```

---

## 🎮 Usage

### Switching Modes

- Default mode on startup is **Media Player mode** (green banner)
- Do the **Point Forward** gesture to switch to **System Control mode** (red banner)
- Do **Point Forward** again to switch back

### Making Gestures

- Hold each gesture clearly in front of the webcam
- Wait for the **hold progress bar** to fill (0.6 seconds)
- The action fires automatically

### Keyboard Shortcuts (Media Player mode)

| Key | Action |
|---|---|
| Space | Pause / Play |
| → Right Arrow | Next video |
| ← Left Arrow | Previous video |
| ↑ Up Arrow | Volume up |
| ↓ Down Arrow | Volume down |
| Escape | Exit fullscreen |

---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Model type | Multi-Layer Perceptron (MLP) |
| Input features | 63 (21 landmarks × 3 coordinates) |
| Hidden layers | 256 → 128 → 64 neurons |
| Activation | ReLU + BatchNorm + Dropout |
| Output | 8 neurons, Softmax |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Test accuracy | 100% |
| Training samples | ~1,618 (balanced, ~200 per class) |

---

## 📊 Gesture Classes

| Class | Description |
|---|---|
| `FIST` | All fingers closed |
| `INDEX_UP` | Only index finger pointing up |
| `OK_SIGN` | Index and thumb touching, others up |
| `OPEN_PALM` | All fingers extended, thumb open |
| `POINT_FORWARD` | Index finger pointing forward horizontally |
| `THREE_FINGERS` | Three fingers up, thumb closed |
| `THUMBS_UP` | Thumb up, fingers closed |
| `V_SIGN` | Index and middle fingers up (peace sign) |

---

## 🔄 Retraining the Model

### Step 1 — Collect data

```bash
python data_collection.py
```

- Press `S` to save a sample
- Press `N` to switch to the next gesture
- Collect ~200 samples per gesture from both hands

### Step 2 — Retrain

```bash
pip install pandas matplotlib
python retrain_model.py
```

The script overwrites `gesture_model.h5` and saves a `training_results.png` graph.

---

## 📦 Requirements

```
opencv-python==4.8.0.76
mediapipe==0.10.7
tensorflow==2.13.0
numpy==1.24.3
scikit-learn==1.3.0
pyautogui==0.9.54
python-vlc==3.0.18122
pandas
matplotlib
```

---

## ⚠️ Known Limitations

- Performance may degrade under very low lighting
- Model trained on single user — accuracy may vary for others
- Only static gestures supported (no swipe or wave)
- Single-hand detection only
- VLC embedding works best on Windows

---

## 🔮 Future Work

- [ ] Mouse control mode
- [ ] Voice feedback (text-to-speech)
- [ ] Dynamic gesture recognition using LSTM
- [ ] Multi-user training data
- [ ] Two-hand simultaneous support
- [ ] TensorFlow Lite for mobile deployment
- [ ] Improved low-light robustness

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9 / 3.10 | Core language |
| TensorFlow / Keras | 2.13.0 | Neural network |
| MediaPipe | 0.10.7 | Hand landmark detection |
| OpenCV | 4.8.0.76 | Camera and frame processing |
| NumPy | 1.24.3 | Array operations |
| Scikit-learn | 1.3.0 | Label encoding, train/test split |
| Pandas | 2.0.0 | CSV data loading |
| Tkinter | Built-in | Graphical user interface |
| python-vlc | 3.0.18122 | VLC media player |
| PyAutoGUI | 0.9.54 | System keyboard control |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👩‍💻 Author

**Hidha**
Integrated M.Sc. Computer Science with Specialization in AI & ML
Nehru Arts and Science College Kanhangad, Kannur University

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://mediapipe.dev/) for the hand landmark detection framework
- [TensorFlow / Keras](https://keras.io/) for the deep learning framework
- [OpenCV](https://opencv.org/) for computer vision utilities
- [python-vlc](https://python-vlc.readthedocs.io/) for VLC media player integration
