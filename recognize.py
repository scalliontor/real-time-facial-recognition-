"""
recognize.py — Ultra-fast tracking + real-time recognition + auto-registration.

Optimizations included:
  - Bipartite matching (SORT algorithm via scipy) for tracking face IDs across frames.
  - Decoupled detection (RetinaFace) from recognition (AuraFace).
  - AuraFace embeddings are extracted ONLY ONCE per track to identify them.
  - If identified, we skip AuraFace for all future frames of that track.
  - If unknown, we wait until steady, then extract AuraFace 5 times to register.
  - Pushes FPS to the limit by avoiding heavy CNN calls on every frame.
"""

import cv2
import numpy as np
import time
import uuid
import os
from scipy.optimize import linear_sum_assignment

from database import get_all_embeddings, add_user
from face_engine import detect_only, extract_features, find_best_match, compute_similarity

FACE_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "registered_faces")

# Tracking and Auto-registration settings
MIN_FACE_WIDTH = 150        # Minimum face bounding box width in pixels
STEADY_THRESHOLD = 25       # Max pixel movement to consider "steady"
STEADY_DURATION = 1.0       # Seconds the face must be steady before capture starts
FRAMES_NEEDED = 5           # Number of embeddings to average for registration
CAPTURE_INTERVAL = 0.3      # Seconds between captures once steady
MATCH_THRESHOLD = 0.5       # Cosine similarity threshold for recognition
TRACK_MAX_DIST = 100        # Max pixel distance to link the same face between frames
TRACK_MAX_AGE = 1.5         # Seconds to keep a lost track alive before deleting


class FastFaceTracker:
    """Tracks a single face across frames using center distance."""

    def __init__(self, track_id, center, bbox):
        self.track_id = track_id
        self.center = center
        self.bbox = bbox
        
        # Tracking timing
        now = time.time()
        self.last_seen = now
        
        # Recognition Phase (AuraFace runs ONCE to set identity)
        self.identified_once = False
        self.identity = None   # Becomes user_id if known
        self.score = 0.0       # Similarity score if known
        
        # Registration Phase (For unknowns only)
        self.steady_start = now
        self.is_steady = False
        self.embeddings = []
        self.last_capture_time = 0

    def update_position(self, center, bbox):
        """Update tracker with new frame coordinates and check steadiness."""
        dx = abs(center[0] - self.center[0])
        dy = abs(center[1] - self.center[1])
        movement = max(dx, dy)

        if movement < STEADY_THRESHOLD:
            elapsed = time.time() - self.steady_start
            self.is_steady = elapsed >= STEADY_DURATION
        else:
            self.steady_start = time.time()
            self.is_steady = False

        self.center = center
        self.bbox = bbox
        self.last_seen = time.time()

    def get_steady_progress(self):
        elapsed = time.time() - self.steady_start
        return min(1.0, elapsed / STEADY_DURATION)


def _match_tracks_to_detections(trackers, detections, max_dist=TRACK_MAX_DIST):
    """
    Hungarian matching between existing trackers and new frame detections.
    Returns: (matches, unmatched_trackers, unmatched_detections)
    """
    if len(trackers) == 0 or len(detections) == 0:
        return [], list(range(len(trackers))), list(range(len(detections)))

    cost_matrix = np.zeros((len(trackers), len(detections)))
    for i, t in enumerate(trackers):
        for j, d_center in enumerate(detections):
            # Euclidean distance
            cost_matrix[i, j] = np.linalg.norm(np.array(t.center) - np.array(d_center))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_trackers = set(range(len(trackers)))
    unmatched_detections = set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_dist:
            matches.append((r, c))
            unmatched_trackers.discard(r)
            unmatched_detections.discard(c)

    return matches, list(unmatched_trackers), list(unmatched_detections)


def start_unified(app, cap, threshold: float = MATCH_THRESHOLD):
    """
    Ultra-fast unified recognition and auto-registration loop.
    """
    os.makedirs(FACE_SAVE_DIR, exist_ok=True)

    db_embeddings = get_all_embeddings()
    print(f"\n[System] Loaded {len(db_embeddings)} registered face(s) from database.")

    # Colors
    COLOR_KNOWN = (0, 255, 0)
    COLOR_UNKNOWN = (0, 0, 255)
    COLOR_TRACKING = (0, 255, 255)
    COLOR_CAPTURING = (255, 165, 0)
    COLOR_INFO = (255, 255, 255)
    COLOR_BG = (0, 0, 0)

    prev_time = time.time()
    fps = 0.0
    registered_count = 0

    trackers = []
    next_track_id = 1

    print(f"\n{'='*55}")
    print(f"  HIGH-FPS TRACKING STAGE")
    print(f"  Detector: RetinaFace (always on)")
    print(f"  Recognizer: AuraFace (skipped for tracked IDs)")
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

        # 1. RUN DETECTION ONLY (Extremely fast, skips 512D embeddings)
        faces = detect_only(app, frame)

        # Build list of centers for Hungarian matching
        det_centers = []
        det_bboxes = []
        for face in faces:
            bbox = face.bbox.astype(int)
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            det_centers.append(center)
            det_bboxes.append(bbox)

        # 2. MATCH TRACKS TO DETECTIONS
        matches, unmatched_t, unmatched_d = _match_tracks_to_detections(trackers, det_centers)

        # Create new trackers for unmatched detections
        for d_idx in unmatched_d:
            new_tracker = FastFaceTracker(next_track_id, det_centers[d_idx], det_bboxes[d_idx])
            trackers.append(new_tracker)
            matches.append((len(trackers)-1, d_idx)) # Instantly add to matches
            next_track_id += 1

        # 3. PROCESS EACH MATCHED TRACKER AND DRAW
        for t_idx, d_idx in matches:
            tracker = trackers[t_idx]
            face = faces[d_idx]
            bbox = det_bboxes[d_idx]
            bbox_width = bbox[2] - bbox[0]

            # Update position (only for frames where we really saw them)
            tracker.update_position(det_centers[d_idx], bbox)

            # Identification Phase (Run AuraFace ONLY ONCE for a new track)
            if not tracker.identified_once:
                embedding = extract_features(app, frame, face)
                user_id, score = find_best_match(embedding, db_embeddings, threshold)
                tracker.identified_once = True
                
                if user_id is not None:
                    # KNOWN PERSON! Cache the identity.
                    tracker.identity = user_id
                    tracker.score = score

            # Drawing Phase based on Tracker State
            if tracker.identity is not None:
                # -> KNOWN (Fast path: drawing without CNN inference!)
                label = f"ID:{tracker.identity} ({tracker.score:.2f}) [Trk:{tracker.track_id}]"
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_KNOWN, 2)

                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = max(bbox[1] - 10, label_size[1] + 10)
                cv2.rectangle(display, (bbox[0], label_y - label_size[1] - 5),
                              (bbox[0] + label_size[0] + 5, label_y + 5), COLOR_KNOWN, -1)
                cv2.putText(display, label, (bbox[0] + 2, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BG, 2)
                
            else:
                # -> UNKNOWN
                if bbox_width < MIN_FACE_WIDTH:
                    # Too far away to register
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_UNKNOWN, 2)
                    label = f"Unknown [Trk:{tracker.track_id}]"
                    cv2.putText(display, label, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_UNKNOWN, 2)
                else:
                    # Close enough. Check steady.
                    if len(tracker.embeddings) >= FRAMES_NEEDED:
                        # REGISTER NEW FACE
                        master_emb = np.mean(tracker.embeddings, axis=0)
                        master_emb = (master_emb / np.linalg.norm(master_emb)).astype(np.float32)
                        new_id = str(uuid.uuid4())[:8]

                        # Save crop
                        h, w = frame.shape[:2]
                        pad = int(bbox_width * 0.2)
                        y1, y2 = max(0, bbox[1]-pad), min(h, bbox[3]+pad)
                        x1, x2 = max(0, bbox[0]-pad), min(w, bbox[2]+pad)
                        crop = frame[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(FACE_SAVE_DIR, f"{new_id}.jpg"), crop)

                        add_user(new_id, master_emb)
                        db_embeddings = get_all_embeddings()
                        registered_count += 1
                        
                        # Cache identity so they are instantly known!
                        tracker.identity = new_id
                        tracker.score = 1.0  # newly registered
                        
                        print(f"  [Auto-Registered] New ID: {new_id}")

                        # Flash effect
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_KNOWN, 4)
                        cv2.putText(display, f"REGISTERED! ID:{new_id}", (bbox[0], bbox[1] - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_KNOWN, 2)

                    elif tracker.is_steady and (now - tracker.last_capture_time) >= CAPTURE_INTERVAL:
                        # CAPTURE EMBEDDING
                        emb = extract_features(app, frame, face)
                        tracker.embeddings.append(emb)
                        tracker.last_capture_time = now
                        
                        count = len(tracker.embeddings)
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_CAPTURING, 3)
                        cv2.putText(display, f"Capturing {count}/{FRAMES_NEEDED}", (bbox[0], bbox[1]-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAPTURING, 2)
                        
                    elif tracker.is_steady:
                        # STEADY WAITING
                        count = len(tracker.embeddings)
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_CAPTURING, 2)
                        cv2.putText(display, f"Capturing {count}/{FRAMES_NEEDED}", (bbox[0], bbox[1]-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAPTURING, 2)

                    else:
                        # HOLD STILL
                        progress = tracker.get_steady_progress()
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_TRACKING, 2)

                        bar_y = bbox[3] + 5
                        cv2.rectangle(display, (bbox[0], bar_y), (bbox[0]+bbox_width, bar_y+8), (50, 50, 50), -1)
                        cv2.rectangle(display, (bbox[0], bar_y), (bbox[0]+int(bbox_width*progress), bar_y+8), COLOR_TRACKING, -1)
                        
                        cv2.putText(display, "Hold still...", (bbox[0], bbox[1] - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TRACKING, 2)

        # 4. CLEANUP STALE TRACKERS
        active_trackers = []
        for t in trackers:
            if (now - t.last_seen) < TRACK_MAX_AGE:
                active_trackers.append(t)
        trackers = active_trackers

        # Info bar
        info_text = f"FPS: {fps:.1f} | Faces: {len(faces)} | DB: {len(db_embeddings)} | New: {registered_count}"
        cv2.rectangle(display, (0, 0), (len(info_text) * 12 + 10, 35), COLOR_BG, -1)
        cv2.putText(display, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)

        cv2.imshow("Optimized Mode — Tracker ON | 'q' quit | 'r' reload", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            db_embeddings = get_all_embeddings()
            print(f"[System] Reloaded database: {len(db_embeddings)} face(s)")

    cv2.destroyAllWindows()
    print(f"[System] Session ended. Registered {registered_count} new face(s).")
