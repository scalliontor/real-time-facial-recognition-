"""
recognize.py — Unified real-time recognition + auto-registration module.

Single mode that:
  - Detects ALL faces in each frame
  - Recognizes known faces (shows ID + confidence in green)
  - Auto-registers unknown faces when they are:
      * Close enough (face width > 150px)
      * Steady (position stable for ~1 second)
      * Collected 5 stable frames for embedding averaging
  - Already-registered faces are never re-registered
  - Tracks multiple faces independently across frames
"""

import cv2
import numpy as np
import time
import uuid
import os

from database import get_all_embeddings, add_user
from face_engine import detect_faces, get_embedding, find_best_match, compute_similarity

FACE_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "registered_faces")

# Auto-registration settings
MIN_FACE_WIDTH = 150        # Minimum face bounding box width in pixels
STEADY_THRESHOLD = 25       # Max pixel movement to consider "steady"
STEADY_DURATION = 1.0       # Seconds the face must be steady before capture starts
FRAMES_NEEDED = 5           # Number of embeddings to average for registration
CAPTURE_INTERVAL = 0.3      # Seconds between captures once steady
MATCH_THRESHOLD = 0.5       # Cosine similarity threshold for recognition
TRACK_DISTANCE = 80         # Max pixel distance to consider same face across frames


class FaceTracker:
    """Tracks an individual unknown face across frames for auto-registration."""

    def __init__(self, center, embedding):
        self.center = center
        self.steady_start = time.time()
        self.is_steady = False
        self.embeddings = []
        self.last_capture_time = 0
        self.last_seen = time.time()
        self.first_embedding = embedding  # Used to check if still same person

    def update_position(self, center):
        """Update face position and check steadiness."""
        dx = abs(center[0] - self.center[0])
        dy = abs(center[1] - self.center[1])
        movement = max(dx, dy)

        if movement < STEADY_THRESHOLD:
            # Still steady
            elapsed = time.time() - self.steady_start
            self.is_steady = elapsed >= STEADY_DURATION
        else:
            # Moved — reset steadiness timer
            self.steady_start = time.time()
            self.is_steady = False

        self.center = center
        self.last_seen = time.time()

    def get_steady_progress(self):
        """Returns 0.0 to 1.0 progress toward steady threshold."""
        elapsed = time.time() - self.steady_start
        return min(1.0, elapsed / STEADY_DURATION)

    def try_capture(self, embedding):
        """Try to capture an embedding if steady and interval elapsed."""
        now = time.time()
        if self.is_steady and (now - self.last_capture_time) >= CAPTURE_INTERVAL:
            self.embeddings.append(embedding)
            self.last_capture_time = now
            return True
        return False

    def is_ready(self):
        """Check if enough frames have been captured for registration."""
        return len(self.embeddings) >= FRAMES_NEEDED

    def get_master_embedding(self):
        """Average and normalize all collected embeddings."""
        master = np.mean(self.embeddings, axis=0)
        master = master / np.linalg.norm(master)
        return master.astype(np.float32)


def _find_closest_tracker(center, trackers, max_dist=TRACK_DISTANCE):
    """Find the tracker closest to the given center point."""
    best_idx = None
    best_dist = max_dist
    for i, tracker in enumerate(trackers):
        dist = max(abs(center[0] - tracker.center[0]), abs(center[1] - tracker.center[1]))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def start_unified(app, cap, threshold: float = MATCH_THRESHOLD):
    """
    Unified real-time recognition + auto-registration.

    Detects all faces, recognizes known ones, and auto-registers
    unknown steady faces. Press 'q' to quit, 'r' to reload DB.

    Args:
        app: Initialized FaceAnalysis object.
        cap: Already-opened cv2.VideoCapture device.
        threshold: Cosine similarity threshold for recognition.
    """
    os.makedirs(FACE_SAVE_DIR, exist_ok=True)

    # Load database
    db_embeddings = get_all_embeddings()
    print(f"\n[System] Loaded {len(db_embeddings)} registered face(s) from database.")

    # Active trackers for unknown faces being observed
    unknown_trackers = []

    # Colors
    COLOR_KNOWN = (0, 255, 0)       # Green — recognized
    COLOR_UNKNOWN = (0, 0, 255)     # Red — unknown (not close/steady enough)
    COLOR_TRACKING = (0, 255, 255)  # Yellow — tracking, hold still
    COLOR_CAPTURING = (255, 165, 0) # Orange — capturing frames
    COLOR_INFO = (255, 255, 255)    # White text
    COLOR_BG = (0, 0, 0)           # Black background

    # FPS tracking
    prev_time = time.time()
    fps = 0.0
    registered_count = 0

    print(f"\n{'='*55}")
    print(f"  UNIFIED MODE — Recognize + Auto-Register")
    print(f"  Threshold: {threshold}")
    print(f"  Close + steady unknowns auto-register")
    print(f"  Press 'q' to quit | 'r' to reload DB")
    print(f"{'='*55}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now

        display = frame.copy()
        faces = detect_faces(app, frame)

        # Track which trackers were matched this frame
        matched_trackers = set()

        for face in faces:
            bbox = face.bbox.astype(int)
            bbox_width = bbox[2] - bbox[0]
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            embedding = get_embedding(face)

            # Step 1: Check if this face is already registered
            user_id, score = find_best_match(embedding, db_embeddings, threshold)

            if user_id is not None:
                # KNOWN FACE — green box with ID
                label = f"ID:{user_id} ({score:.2f})"
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_KNOWN, 2)

                # Label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = max(bbox[1] - 10, label_size[1] + 10)
                cv2.rectangle(display, (bbox[0], label_y - label_size[1] - 5),
                              (bbox[0] + label_size[0] + 5, label_y + 5), COLOR_KNOWN, -1)
                cv2.putText(display, label, (bbox[0] + 2, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BG, 2)
                continue

            # Step 2: UNKNOWN FACE — check if close enough for registration
            if bbox_width < MIN_FACE_WIDTH:
                # Too far — just show red box
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_UNKNOWN, 2)
                label = "Unknown"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(bbox[1] - 10, label_size[1] + 10)
                cv2.rectangle(display, (bbox[0], label_y - label_size[1] - 5),
                              (bbox[0] + label_size[0] + 5, label_y + 5), COLOR_UNKNOWN, -1)
                cv2.putText(display, label, (bbox[0] + 2, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BG, 2)
                continue

            # Step 3: Close enough — find or create a tracker for this face
            tracker_idx = _find_closest_tracker(center, unknown_trackers)

            if tracker_idx is not None:
                tracker = unknown_trackers[tracker_idx]
                matched_trackers.add(tracker_idx)

                # Verify it's still the same person (embedding similarity)
                sim = compute_similarity(embedding, tracker.first_embedding)
                if sim < 0.3:
                    # Different person took this position — reset tracker
                    unknown_trackers[tracker_idx] = FaceTracker(center, embedding)
                    tracker = unknown_trackers[tracker_idx]

                tracker.update_position(center)

                if tracker.is_ready():
                    # REGISTER THIS FACE
                    master_embedding = tracker.get_master_embedding()
                    new_id = str(uuid.uuid4())[:8]

                    # Save face crop
                    h, w = frame.shape[:2]
                    pad = int(bbox_width * 0.2)
                    y1, y2 = max(0, bbox[1] - pad), min(h, bbox[3] + pad)
                    x1, x2 = max(0, bbox[0] - pad), min(w, bbox[2] + pad)
                    face_crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(FACE_SAVE_DIR, f"{new_id}.jpg"), face_crop)

                    # Save to database
                    add_user(new_id, master_embedding)
                    db_embeddings = get_all_embeddings()  # Reload
                    registered_count += 1

                    # Remove this tracker
                    unknown_trackers.pop(tracker_idx)
                    matched_trackers.discard(tracker_idx)

                    print(f"  [Auto-Registered] New ID: {new_id} (total: {len(db_embeddings)})")

                    # Flash green
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_KNOWN, 4)
                    label = f"REGISTERED! ID:{new_id}"
                    cv2.putText(display, label, (bbox[0], bbox[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_KNOWN, 2)

                elif tracker.try_capture(embedding):
                    # Just captured a frame
                    count = len(tracker.embeddings)
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_CAPTURING, 3)
                    label = f"Capturing {count}/{FRAMES_NEEDED}"
                    cv2.putText(display, label, (bbox[0], bbox[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAPTURING, 2)

                elif tracker.is_steady:
                    # Steady, waiting for next capture interval
                    count = len(tracker.embeddings)
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_CAPTURING, 2)
                    label = f"Capturing {count}/{FRAMES_NEEDED}"
                    cv2.putText(display, label, (bbox[0], bbox[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAPTURING, 2)

                else:
                    # Not steady yet — show progress bar
                    progress = tracker.get_steady_progress()
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_TRACKING, 2)

                    bar_w = bbox_width
                    bar_x = bbox[0]
                    bar_y = bbox[3] + 5
                    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50, 50, 50), -1)
                    cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 8), COLOR_TRACKING, -1)

                    label = "Hold still..."
                    cv2.putText(display, label, (bbox[0], bbox[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TRACKING, 2)
            else:
                # New unknown face — create tracker
                new_tracker = FaceTracker(center, embedding)
                unknown_trackers.append(new_tracker)
                matched_trackers.add(len(unknown_trackers) - 1)

                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_TRACKING, 2)
                label = "Detecting..."
                cv2.putText(display, label, (bbox[0], bbox[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TRACKING, 2)

        # Clean up stale trackers (not seen for > 2 seconds)
        unknown_trackers = [t for i, t in enumerate(unknown_trackers)
                           if (now - t.last_seen) < 2.0]

        # Info bar
        info_text = f"FPS: {fps:.1f} | Faces: {len(faces)} | DB: {len(db_embeddings)} | New: {registered_count}"
        cv2.rectangle(display, (0, 0), (len(info_text) * 12 + 10, 35), COLOR_BG, -1)
        cv2.putText(display, info_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)

        cv2.imshow("Unified Mode — Auto Recognize + Register | 'q' quit | 'r' reload", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            db_embeddings = get_all_embeddings()
            print(f"[System] Reloaded database: {len(db_embeddings)} face(s)")

    cv2.destroyAllWindows()
    print(f"[System] Session ended. Registered {registered_count} new face(s).")
