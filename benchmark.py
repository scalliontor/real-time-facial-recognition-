import cv2
import numpy as np
import time
import uuid
import os
import psutil
import subprocess
from scipy.optimize import linear_sum_assignment

from database import get_all_embeddings, add_user, init_db
from face_engine import init_face_app, detect_only, extract_features, find_best_match, compute_similarity

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "report")
os.makedirs(REPORT_DIR, exist_ok=True)
FACE_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "registered_faces")

# Tracking settings
MIN_FACE_WIDTH = 150
STEADY_THRESHOLD = 25
STEADY_DURATION = 1.0
FRAMES_NEEDED = 5
CAPTURE_INTERVAL = 0.3
MATCH_THRESHOLD = 0.5
RECOG_FRAME_INTERVAL = 5
TRACK_MAX_DIST = 100
TRACK_MAX_AGE = 1.5

class FastFaceTracker:
    def __init__(self, track_id, center, bbox):
        self.track_id = track_id
        self.center = center
        self.bbox = bbox
        self.last_seen = time.time()
        self.identity = None
        self.score = 0.0
        self.frames_since_recog = RECOG_FRAME_INTERVAL
        self.steady_start = time.time()
        self.is_steady = False
        self.embeddings = []
        self.last_capture_time = 0

    def needs_recognition(self):
        if self.identity is not None:
            return False
        return self.frames_since_recog >= RECOG_FRAME_INTERVAL

    def update_position(self, center, bbox):
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

def _match_tracks(trackers, detections, max_dist=TRACK_MAX_DIST):
    if len(trackers) == 0 or len(detections) == 0:
        return [], list(range(len(trackers))), list(range(len(detections)))
    cost_matrix = np.zeros((len(trackers), len(detections)))
    for i, t in enumerate(trackers):
        for j, d_center in enumerate(detections):
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


def get_gpu_metrics():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        parts = output.strip().split(',')
        return float(parts[0]), float(parts[1])
    except:
        return 0.0, 0.0

def run_benchmark():
    print("========== BENCHMARK SUITE ==========")
    print("Initializing...")
    init_db()
    app = init_face_app()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(REPORT_DIR, 'demo.mp4'), fourcc, 20.0, (1280, 720))

    db_embeddings = get_all_embeddings()
    trackers = []
    next_track_id = 1
    
    # Metrics
    detect_lat = []
    extract_lat = []
    match_lat = []
    fps_history = []
    faces_history = []
    cpu_history = []
    ram_history = []
    gpu_util_history = []
    gpu_mem_history = []
    
    # Flags for screenshots
    saved_unknown = False
    saved_registering = False
    saved_registered = False
    saved_recognized = False

    COLOR_KNOWN = (0, 255, 0)
    COLOR_UNKNOWN = (0, 0, 255)
    COLOR_TRACKING = (0, 255, 255)
    COLOR_CAPTURING = (255, 165, 0)
    COLOR_INFO = (255, 255, 255)
    COLOR_BG = (0, 0, 0)

    print("Starting capture for 300 frames (approx 15 seconds)... Please interact with the camera.")
    
    frame_count = 0
    max_frames = 300
    prev_time = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now
        display = frame.copy()
        
        # Detection
        t0 = time.time()
        faces = detect_only(app, frame)
        detect_lat.append((time.time() - t0) * 1000)
        
        det_centers = []
        det_bboxes = []
        for face in faces:
            bbox = face.bbox.astype(int)
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            det_centers.append(center)
            det_bboxes.append(bbox)

        matches, unmatched_t, unmatched_d = _match_tracks(trackers, det_centers)

        for d_idx in unmatched_d:
            trackers.append(FastFaceTracker(next_track_id, det_centers[d_idx], det_bboxes[d_idx]))
            matches.append((len(trackers)-1, d_idx))
            next_track_id += 1

        state_this_frame = None

        for t_idx, d_idx in matches:
            tracker = trackers[t_idx]
            face = faces[d_idx]
            bbox = det_bboxes[d_idx]
            bbox_width = bbox[2] - bbox[0]

            tracker.update_position(det_centers[d_idx], bbox)
            tracker.frames_since_recog += 1

            if tracker.needs_recognition():
                t1 = time.time()
                embedding = extract_features(app, frame, face)
                extract_lat.append((time.time() - t1) * 1000)
                
                t2 = time.time()
                user_id, score = find_best_match(embedding, db_embeddings, MATCH_THRESHOLD)
                match_lat.append((time.time() - t2) * 1000)
                
                tracker.frames_since_recog = 0
                if user_id is not None:
                    tracker.identity = user_id
                    tracker.score = score
                    tracker.embeddings = []
                    state_this_frame = 'recognized'

            if tracker.identity is not None:
                state_this_frame = 'recognized'
                label = f"ID:{tracker.identity} ({tracker.score:.2f}) [Trk:{tracker.track_id}]"
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_KNOWN, 2)
                cv2.putText(display, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_KNOWN, 2)
            else:
                if bbox_width < MIN_FACE_WIDTH:
                    state_this_frame = 'unknown'
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_UNKNOWN, 2)
                    cv2.putText(display, "Unknown", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_UNKNOWN, 2)
                else:
                    if len(tracker.embeddings) >= FRAMES_NEEDED:
                        master_emb = np.mean(tracker.embeddings, axis=0)
                        master_emb = (master_emb / np.linalg.norm(master_emb)).astype(np.float32)
                        new_id = str(uuid.uuid4())[:8]
                        add_user(new_id, master_emb)
                        db_embeddings = get_all_embeddings()
                        tracker.identity = new_id
                        tracker.score = 1.0
                        state_this_frame = 'registered'
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_KNOWN, 4)
                        cv2.putText(display, f"REGISTERED! ID:{new_id}", (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_KNOWN, 2)
                    elif tracker.is_steady and (now - tracker.last_capture_time) >= CAPTURE_INTERVAL:
                        emb = extract_features(app, frame, face)
                        tracker.embeddings.append(emb)
                        tracker.last_capture_time = now
                        state_this_frame = 'registering'
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_CAPTURING, 3)
                        cv2.putText(display, f"Capturing {len(tracker.embeddings)}/{FRAMES_NEEDED}", (bbox[0], bbox[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAPTURING, 2)
                    elif tracker.is_steady:
                        state_this_frame = 'registering'
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_CAPTURING, 2)
                        cv2.putText(display, f"Capturing {len(tracker.embeddings)}/{FRAMES_NEEDED}", (bbox[0], bbox[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAPTURING, 2)
                    else:
                        state_this_frame = 'tracking'
                        progress = tracker.get_steady_progress()
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_TRACKING, 2)
                        bar_y = bbox[3] + 5
                        cv2.rectangle(display, (bbox[0], bar_y), (bbox[0]+int(bbox_width*progress), bar_y+8), COLOR_TRACKING, -1)
                        cv2.putText(display, "Hold still...", (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TRACKING, 2)

        trackers = [t for t in trackers if (now - t.last_seen) < TRACK_MAX_AGE]

        info_text = f"FPS: {fps:.1f} | Faces: {len(faces)} | DB: {len(db_embeddings)}"
        cv2.putText(display, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)

        # Save screenshots
        if state_this_frame == 'unknown' and not saved_unknown:
            cv2.imwrite(os.path.join(REPORT_DIR, 'demo_unknown.jpg'), display)
            saved_unknown = True
        elif state_this_frame == 'registering' and not saved_registering:
            cv2.imwrite(os.path.join(REPORT_DIR, 'demo_registering.jpg'), display)
            saved_registering = True
        elif state_this_frame == 'registered' and not saved_registered:
            cv2.imwrite(os.path.join(REPORT_DIR, 'demo_registered.jpg'), display)
            saved_registered = True
        elif state_this_frame == 'recognized' and not saved_recognized:
            cv2.imwrite(os.path.join(REPORT_DIR, 'demo_recognized.jpg'), display)
            saved_recognized = True

        out_video.write(display)

        # Metrics collection
        if fps > 0 and len(faces) >= 0:
            fps_history.append(fps)
            faces_history.append(len(faces))
        
        if frame_count % 10 == 0:
            cpu_history.append(psutil.cpu_percent())
            ram_history.append(psutil.virtual_memory().used / (1024**2)) # MB
            gpu_u, gpu_m = get_gpu_metrics()
            gpu_util_history.append(gpu_u)
            gpu_mem_history.append(gpu_m)

        frame_count += 1
        cv2.imshow("Benchmarking...", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    print("\n\n========== BENCHMARK RESULTS ==========")
    
    
    avg_det = np.mean(detect_lat) if detect_lat else 0
    avg_ext = np.mean(extract_lat) if extract_lat else 0
    avg_mat = np.mean(match_lat) if match_lat else 0
    
    fps_by_face = {}
    for i in range(len(fps_history)):
        fc = faces_history[i]
        if fc not in fps_by_face:
            fps_by_face[fc] = []
        fps_by_face[fc].append(fps_history[i])
        
    report_path = os.path.join(REPORT_DIR, "benchmark_metrics.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Results\n\n")
        
        f.write("## Latency\n")
        f.write(f"- **Detection:** {avg_det:.2f} ms\n")
        f.write(f"- **Extraction:** {avg_ext:.2f} ms\n")
        f.write(f"- **Matching:** {avg_mat:.2f} ms\n")
        f.write(f"- **End-to-End per recognized frame:** {(avg_det + avg_ext + avg_mat):.2f} ms\n\n")
        
        f.write("## FPS by Face Count\n")
        for faces in sorted(fps_by_face.keys()):
            arr = fps_by_face[faces]
            if len(arr) > 0:
                f.write(f"- **{faces} Face(s):** Avg {np.mean(arr):.1f} | Min {np.min(arr):.1f} | Max {np.max(arr):.1f}\n")
        
        f.write("\n## Resources (Avg)\n")
        f.write(f"- **CPU:** {np.mean(cpu_history):.1f} %\n")
        f.write(f"- **RAM:** {np.mean(ram_history):.1f} MB\n")
        f.write(f"- **GPU Util:** {np.mean(gpu_util_history):.1f} %\n")
        f.write(f"- **GPU VRAM:** {np.mean(gpu_mem_history):.1f} MB\n")

    print(f"Report saved to {report_path}")

if __name__ == '__main__':
    run_benchmark()
