# Real-Time Facial Recognition System

An ultra-fast, real-time facial recognition and auto-registration system built with Python, InsightFace, AuraFace-v1, and a custom SORT-based tracker. It uses OpenCV for video capture and SQLite for storing embeddings.

## Key Features

- **Ultra-Fast Tracking**: Uses a decoupled approach to maximize FPS. The heavy AuraFace recognition model (512-D CNN) is only run *once* per tracked person. A lightweight RetinaFace detector and a custom SORT (Hungarian matching via SciPy) tracker keep track of face IDs in real-time, bypassing heavy inference for 99% of frames.
- **Auto-Registration**: Unknown faces that are close enough (configurable width) and steady for 1 second are automatically captured and averaged over 5 frames. The master embedding and a cropped face image are saved for future use.
- **Unified Mode**: A single, seamless interface that handles both recognition of known individuals and background auto-registration of unknown individuals simultaneously.
- **Multiple People Support**: Independently detects, tracks, and registers multiple faces in the same frame.
- **GPU Acceleration**: Built-in support for ONNXRuntime `CUDAExecutionProvider` to leverage NVIDIA GPUs for maximum performance.

## Prerequisites

- **Python**: 3.9 - 3.12 (Tested on Windows).
- **Webcam**: A functional webcam is required for real-time capture.
- **CUDA & cuDNN**: If using an NVIDIA GPU, ensure CUDA Toolkit and cuDNN are installed and added to your system PATH to utilize `onnxruntime-gpu`.

## Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:scalliontor/real-time-facial-recognition-.git
   cd real-time-facial-recognition-
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # Activate it:
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

On Windows, you can simply run the provided batch script which handles virtual environment activation and sets the cuDNN path (modify the path inside `run.bat` if your cuDNN is located elsewhere):

```cmd
run.bat
```

Alternatively, run the script directly via Python:
```bash
python main.py
```

### Main Menu Options

When you start the application, you will be presented with the following menu:

```
┌─────────────────────────────────────────┐
│              MAIN MENU                  │
├─────────────────────────────────────────┤
│  1) Start (Recognize + Auto-Register)   │
│  2) List Registered Users               │
│  3) Delete a User                       │
│  4) Clear All Users                     │
│  5) Exit                                │
└─────────────────────────────────────────┘
```

#### Option 1: Start (Unified Mode)
- **Known Faces**: Highlighted with a **Green** bounding box showing their generated ID and confidence score.
- **Unknown Faces (Too Far)**: Highlighted with a **Red** bounding box.
- **Unknown Faces (Close & Steady)**: Highlighted with a **Yellow** progress bar. The system requires the person to hold steady for 1 second.
- **Capturing Stage**: Highlighted with an **Orange** bounding box showing the current capture progress (e.g., 3/5).
- **Registered**: Flashes Green with a "REGISTERED!" label once complete.

#### Options 2-4: User Management
Allows you to list all registered UUIDs, delete an individual user (removes their database entry and saved face crop), or clear the entire database.

## Architecture

* `main.py`: The CLI entry point and menu system. Handles webcam initialization.
* `face_engine.py`: Wrapper for InsightFace. Handles downloading the AuraFace-v1 model, detecting faces (`detect_only`), extracting 512-D embeddings (`extract_features`), and computing cosine similarity.
* `recognize.py`: The core unified loop. Implements the `FastFaceTracker` class to handle tracking states (steady, capturing, identified) using the SciPy `linear_sum_assignment` function.
* `database.py`: SQLite wrapper for storing user IDs and their 512-D normalized embeddings as BLOBs.

## Data Storage
- SQLite database: Generated securely inside `data/faces.db`
- Cropped User Faces: Saved as `.jpg` inside `data/registered_faces/`. These serve as visual references and allow re-extraction with new models without re-registering users.
