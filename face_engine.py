"""
face_engine.py — InsightFace + AuraFace-v1 wrapper for face detection and recognition.

Downloads AuraFace-v1 from HuggingFace on first run.
Uses CUDAExecutionProvider (GPU) with automatic fallback to CPUExecutionProvider.
"""

import os
import numpy as np

# Set the root directory for models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "auraface")


def download_model():
    """Download AuraFace-v1 model from HuggingFace if not already present."""
    if os.path.exists(MODEL_DIR) and any(
        f.endswith(".onnx") for f in os.listdir(MODEL_DIR)
    ):
        print("[Engine] AuraFace-v1 model already downloaded.")
        return

    print("[Engine] Downloading AuraFace-v1 from HuggingFace...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        "fal/AuraFace-v1",
        local_dir=MODEL_DIR,
    )
    print("[Engine] AuraFace-v1 download complete!")


def init_face_app():
    """
    Initialize the InsightFace FaceAnalysis pipeline with AuraFace-v1.

    Returns:
        FaceAnalysis app object ready for inference.
    """
    # Ensure model is downloaded
    download_model()

    from insightface.app import FaceAnalysis

    # Determine available providers
    providers = []
    try:
        import onnxruntime

        available = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            print("[Engine] Using GPU (CUDAExecutionProvider)")
        else:
            print("[Engine] CUDA not available, using CPU")
    except Exception:
        print("[Engine] Could not query providers, defaulting to CPU")

    providers.append("CPUExecutionProvider")

    # Initialize FaceAnalysis with AuraFace model
    app = FaceAnalysis(
        name="auraface",
        root=ROOT_DIR,
        providers=providers,
    )

    # Prepare with detection size (640x640 is standard, good balance of speed/accuracy)
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("[Engine] Face analysis pipeline ready!")
    return app


def detect_faces(app, frame):
    """
    Detect and analyze everything (detection + recognition) for all faces.

    Args:
        app: Initialized FaceAnalysis object.
        frame: BGR numpy array from OpenCV.

    Returns:
        List of face objects with bounding boxes and embeddings.
    """
    faces = app.get(frame)
    return faces


def detect_only(app, frame):
    """
    Detects faces WITHOUT running the heavy recognition model.
    Use this for high-FPS tracking.
    """
    bboxes, kpss = app.det_model.detect(frame, max_num=0, metric='default')
    if bboxes.shape[0] == 0:
        return []
        
    faces = []
    from insightface.app.common import Face
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        faces.append(face)
    return faces


def extract_features(app, frame, face) -> np.ndarray:
    """
    Runs the recognition model on a specifically detected face to extract its 512-D embedding.
    Modifies the face object in-place and returns the embedding.
    """
    for taskname, model in app.models.items():
        if taskname == 'detection':
            continue
        model.get(frame, face)
    return face.normed_embedding.astype(np.float32)


def get_embedding(face) -> np.ndarray:
    """
    Extract the normalized 512-D embedding from a detected face.

    Args:
        face: Face object from InsightFace.

    Returns:
        512-D normalized numpy array (float32).
    """
    return face.normed_embedding.astype(np.float32)


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two face embeddings.

    Args:
        embedding1: 512-D normalized embedding.
        embedding2: 512-D normalized embedding.

    Returns:
        Cosine similarity score (0.0 to 1.0). Higher = more similar.
    """
    # Since embeddings are already normalized, cosine similarity = dot product
    return float(np.dot(embedding1, embedding2))


def find_best_match(query_embedding: np.ndarray, db_embeddings: list, threshold: float = 0.5):
    """
    Find the best matching face in the database.

    Args:
        query_embedding: 512-D embedding of the query face.
        db_embeddings: List of (user_id, embedding) tuples from database.
        threshold: Minimum similarity score to consider a match.

    Returns:
        Tuple (user_id, score) of the best match, or (None, 0.0) if no match.
    """
    best_match = (None, 0.0)

    for user_id, db_embedding in db_embeddings:
        score = compute_similarity(query_embedding, db_embedding)
        if score > best_match[1]:
            best_match = (user_id, score)

    if best_match[1] >= threshold:
        return best_match
    else:
        return (None, 0.0)
