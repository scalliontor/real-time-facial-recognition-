"""
register.py — Auto face enrollment module.

Automatically captures frames when a face is detected as:
  - Close enough (bounding box > 150px wide)
  - Steady (face position stable for ~1 second)
  - Only one person in frame

Averages multiple embeddings into a robust "master embedding"
and saves to the database. Also saves cropped face image.
"""

import cv2
import numpy as np
import os
import uuid
import time

from database import add_user
from face_engine import detect_faces, get_embedding

FACE_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "registered_faces")

# Auto-capture settings
MIN_FACE_WIDTH = 150        # Minimum face bounding box width in pixels
STEADY_THRESHOLD = 20       # Max pixel movement to consider "steady"
STEADY_DURATION = 1.0       # Seconds the face must be steady before capture
FRAMES_NEEDED = 5           # Number of embeddings to average
CAPTURE_INTERVAL = 0.3      # Seconds between captures once steady


def register_face(app, cap):
    """
    Register a new user via webcam with auto-capture.

    Shows live preview. When a face is close and steady enough,
    automatically captures frames without user intervention.
    No name needed — just an auto-generated ID.

    Args:
        app: Initialized FaceAnalysis object.
        cap: Already-opened cv2.VideoCapture device (shared).

    Returns:
        user_id if successful, None if cancelled.
    """
    os.makedirs(FACE_SAVE_DIR, exist_ok=True)

    # Generate unique ID upfront
    user_id = str(uuid.uuid4())[:8]

    collected_embeddings = []
    best_face_image = None
    best_face_size = 0

    # Steadiness tracking
    last_bbox_center = None
    steady_start_time = None
    last_capture_time = 0
    is_steady = False

    print(f"\n{'='*50}")
    print(f"  AUTO-REGISTERING — ID: {user_id}")
    print(f"  Look at the camera and hold still.")
    print(f"  Frames auto-capture when face is steady.")
    print(f"  Press 'q' to cancel.")
    print(f"{'='*50}\n")

    while len(collected_embeddings) < FRAMES_NEEDED:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        faces = detect_faces(app, frame)
        now = time.time()

        # Status bar
        cv2.rectangle(display, (0, 0), (640, 90), (0, 0, 0), -1)
        status_text = f"Captured: {len(collected_embeddings)}/{FRAMES_NEEDED}"
        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if len(faces) == 0:
            cv2.putText(display, "No face detected - step into frame", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            last_bbox_center = None
            steady_start_time = None
            is_steady = False

        elif len(faces) > 1:
            cv2.putText(display, "Multiple faces! Only 1 person please", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            last_bbox_center = None
            steady_start_time = None
            is_steady = False

        else:
            # Exactly one face
            face = faces[0]
            bbox = face.bbox.astype(int)
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            if bbox_width < MIN_FACE_WIDTH:
                # Too far away
                cv2.putText(display, "Too far! Step closer", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)
                last_bbox_center = None
                steady_start_time = None
                is_steady = False
            else:
                # Face is close enough — check steadiness
                if last_bbox_center is not None:
                    dx = abs(center_x - last_bbox_center[0])
                    dy = abs(center_y - last_bbox_center[1])
                    movement = max(dx, dy)

                    if movement < STEADY_THRESHOLD:
                        # Face is steady
                        if steady_start_time is None:
                            steady_start_time = now

                        steady_elapsed = now - steady_start_time

                        if steady_elapsed >= STEADY_DURATION:
                            is_steady = True
                        else:
                            is_steady = False
                            # Show progress bar for steadiness
                            progress = steady_elapsed / STEADY_DURATION
                            bar_width = 300
                            bar_x = 10
                            bar_y = 75
                            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + 12), (50, 50, 50), -1)
                            cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 12), (0, 255, 255), -1)
                            cv2.putText(display, "Hold still...", (10, 65),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # Face moved too much
                        steady_start_time = None
                        is_steady = False
                else:
                    steady_start_time = now
                    is_steady = False

                last_bbox_center = (center_x, center_y)

                # Auto-capture when steady
                if is_steady and (now - last_capture_time) >= CAPTURE_INTERVAL:
                    embedding = get_embedding(face)
                    collected_embeddings.append(embedding)
                    last_capture_time = now

                    # Green flash effect
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
                    cv2.putText(display, f"AUTO-CAPTURED! ({len(collected_embeddings)}/{FRAMES_NEEDED})", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Save the largest/best face crop
                    if bbox_width > best_face_size:
                        best_face_size = bbox_width
                        h, w = frame.shape[:2]
                        pad = int(bbox_width * 0.2)
                        y1 = max(0, bbox[1] - pad)
                        y2 = min(h, bbox[3] + pad)
                        x1 = max(0, bbox[0] - pad)
                        x2 = min(w, bbox[2] + pad)
                        best_face_image = frame[y1:y2, x1:x2].copy()

                    print(f"  [Register] Frame {len(collected_embeddings)}/{FRAMES_NEEDED} auto-captured")

                elif is_steady:
                    # Steady, waiting for next capture interval
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(display, "STEADY - Capturing...", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Close but not steady yet
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)

        cv2.imshow("Auto Registration - Press 'q' to cancel", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[Register] Registration cancelled by user.")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Average all collected embeddings
    master_embedding = np.mean(collected_embeddings, axis=0)

    # Re-normalize (crucial for cosine similarity)
    master_embedding = master_embedding / np.linalg.norm(master_embedding)
    master_embedding = master_embedding.astype(np.float32)

    # Save the best face crop
    if best_face_image is not None:
        face_path = os.path.join(FACE_SAVE_DIR, f"{user_id}.jpg")
        cv2.imwrite(face_path, best_face_image)
        print(f"[Register] Face image saved to {face_path}")

    # Save to database
    add_user(user_id, master_embedding)

    print(f"\n{'='*50}")
    print(f"  SUCCESS!")
    print(f"  ID: {user_id}")
    print(f"  Embedding: {master_embedding.shape[0]}-D vector")
    print(f"{'='*50}\n")

    return user_id
