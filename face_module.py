


import cv2
import numpy as np
import os
import time
import threading
import tempfile
from dataclasses import dataclass, field
from typing import Optional

# ── DeepFace — graceful import ──────────────────────────────────────────
try:
    from deepface import DeepFace
    FACE_LIB_OK = True
except ImportError:
    DeepFace    = None
    FACE_LIB_OK = False

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
DATASET_DIR    = "dataset"          # root folder containing one subfolder per person
FACE_SKIP      = 3                  # run DeepFace every N frames
UNKNOWN_LABEL  = "UNKNOWN"
TRACK_TIMEOUT  = 3.0                # seconds before a face track expires
MIN_FACE_SIZE  = 60                 # minimum crop dimension to attempt recognition

# DeepFace model — balance between speed and accuracy:
#   "VGG-Face"   → high accuracy, larger model
#   "Facenet"    → excellent accuracy + speed  ← default
#   "Facenet512" → best accuracy, slower
#   "ArcFace"    → fast, good accuracy
DF_MODEL    = "Facenet"

# Distance metric: "cosine" | "euclidean" | "euclidean_l2"
DF_METRIC   = "cosine"

# Cosine distance threshold (lower = stricter; good range: 0.30–0.50)
DF_THRESHOLD = 0.40

# Face detector backend: "opencv" (fastest) | "ssd" | "retinaface" (best)
DF_DETECTOR  = "opencv"


# ─────────────────────────────────────────────────────────
# DATA CLASSES  (unchanged from face_recognition version)
# ─────────────────────────────────────────────────────────
@dataclass
class FaceTrack:
    """Persists a recognized identity across frames."""
    track_id:   int
    name:       str
    confidence: float
    bbox:       tuple           # (x1, y1, x2, y2)
    last_seen:  float = field(default_factory=time.time)
    authorized: bool  = False

    def update(self, name: str, confidence: float, bbox: tuple):
        self.name       = name
        self.confidence = confidence
        self.bbox       = bbox
        self.last_seen  = time.time()
        self.authorized = (name != UNKNOWN_LABEL)


@dataclass
class RecognitionResult:
    """Per-face result — same shape as old face_recognition version."""
    name:       str
    confidence: float           # 0.0–1.0
    authorized: bool
    bbox:       tuple           # (x1, y1, x2, y2) in original frame coords
    track_id:   int = -1


# ─────────────────────────────────────────────────────────
# DEEPFACE RECOGNIZER  (drop-in replacement)
# ─────────────────────────────────────────────────────────
class FaceRecognizer:
    """
    DeepFace-based drop-in replacement for the old face_recognition recognizer.

    Key changes vs old version:
      • No encodings.pkl — DeepFace.find() searches dataset/ folder directly
      • load_encodings() validates the dataset folder instead of loading .pkl
      • process_frame() passes the YOLO crop to DeepFace.find()
      • enforce_detection=False — works even on borderline crops
      • All tracking, frame-skip, and draw logic unchanged
    """

    def __init__(self,
                 tolerance:  float = DF_THRESHOLD,
                 skip:       int   = FACE_SKIP,
                 model_name: str   = DF_MODEL,
                 metric:     str   = DF_METRIC,
                 detector:   str   = DF_DETECTOR,
                 dataset:    str   = DATASET_DIR):

        self.tolerance        = tolerance
        self.skip             = skip
        self.model_name       = model_name
        self.metric           = metric
        self.detector         = detector
        self.dataset          = dataset

        # True once dataset/ is validated as non-empty
        self.encodings_loaded = False
        self._known_people:   list[str] = []
        self._lock            = threading.Lock()

        # Face tracking state (identical to old version)
        self._tracks:       dict[int, FaceTrack] = {}
        self._next_id:      int = 0
        self._last_results: list[RecognitionResult] = []

    # ──────────────────────────────────────────────────
    # VALIDATE DATASET  (replaces load_encodings .pkl)
    # ──────────────────────────────────────────────────
    def load_encodings(self, path: str = DATASET_DIR) -> bool:
        """
        Validates the dataset/ directory.
        DeepFace reads images from disk at recognition time — no pre-encoding.

        Args:
            path: passed for API compatibility; self.dataset takes priority.
        Returns:
            True if at least one person folder with images exists.
        """
        if not FACE_LIB_OK:
            return False

        root = self.dataset or path

        if not os.path.isdir(root):
            print(f"[FaceRecognizer] Dataset not found: {root}")
            self.encodings_loaded = False
            return False

        supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        people    = []

        for name in sorted(os.listdir(root)):
            person_dir = os.path.join(root, name)
            if not os.path.isdir(person_dir):
                continue
            imgs = [f for f in os.listdir(person_dir)
                    if os.path.splitext(f)[1].lower() in supported]
            if imgs:
                people.append(name)

        with self._lock:
            self._known_people    = people
            self.encodings_loaded = len(people) > 0

        if people:
            print(f"[FaceRecognizer] Dataset OK — {len(people)} person(s): {people}")
        else:
            print(f"[FaceRecognizer] Dataset empty: {root}")

        return self.encodings_loaded

    def encoding_count(self) -> int:
        """Count total images in the dataset (used for sidebar display)."""
        if not self.encodings_loaded:
            return 0
        total     = 0
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for person in self._known_people:
            p = os.path.join(self.dataset, person)
            if os.path.isdir(p):
                total += sum(
                    1 for f in os.listdir(p)
                    if os.path.splitext(f)[1].lower() in supported
                )
        return total

    def known_people(self) -> list[str]:
        with self._lock:
            return list(self._known_people)

    # ──────────────────────────────────────────────────
    # CORE: DEEPFACE RECOGNITION ON ONE CROP
    # ──────────────────────────────────────────────────
    def _recognize_crop(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Call DeepFace.find() on a BGR face crop.
        Returns (person_name, confidence) or (UNKNOWN_LABEL, score).

        DeepFace.find():
          - Accepts numpy array (BGR) directly
          - Searches all images under self.dataset/
          - Returns a list of DataFrames (one per face found in img_path)
          - Each row = one matched image from the database
          - Sorted by distance (ascending) by default
        """
        try:
            df_results = DeepFace.find(
                img_path         = crop_bgr,
                db_path          = self.dataset,
                model_name       = self.model_name,
                distance_metric  = self.metric,
                detector_backend = self.detector,
                enforce_detection= False,   # don't raise on failed detection
                silent           = True,    # suppress DeepFace console logs
            )

            # df_results is a list — index 0 = results for the first face in crop
            if not df_results or df_results[0].empty:
                return UNKNOWN_LABEL, 0.0

            df = df_results[0]

            # ── Find the distance column (varies by model/metric) ──────
            dist_col = next(
                (col for col in df.columns if "distance" in col.lower()),
                None
            )
            if dist_col is None:
                return UNKNOWN_LABEL, 0.0

            # Sort ascending — best match first
            df       = df.sort_values(dist_col).reset_index(drop=True)
            best_row = df.iloc[0]
            best_dist = float(best_row[dist_col])

            # ── Apply threshold ────────────────────────────────────────
            if best_dist > self.tolerance:
                return UNKNOWN_LABEL, round(max(0.0, 1.0 - best_dist), 3)

            # ── Extract person name from matched file path ─────────────
            # Path looks like:  dataset/PersonName/photo.jpg
            identity  = str(best_row.get("identity", ""))
            if not identity:
                return UNKNOWN_LABEL, 0.0

            person_name = os.path.basename(os.path.dirname(identity))
            confidence  = round(max(0.0, min(1.0, 1.0 - best_dist)), 3)
            return person_name, confidence

        except Exception:
            # Silently swallow exceptions (no face, model error, etc.)
            # For debugging: uncomment below
            # import traceback; traceback.print_exc()
            return UNKNOWN_LABEL, 0.0

    # ──────────────────────────────────────────────────
    # PROCESS FRAME  (public — identical signature)
    # ──────────────────────────────────────────────────
    def process_frame(self,
                      frame:     np.ndarray,
                      frame_num: int,
                      roi:       Optional[tuple] = None) -> list[RecognitionResult]:
        """
        Recognize faces in one frame.

        Args:
            frame:     BGR frame from OpenCV
            frame_num: current frame counter (for frame-skip logic)
            roi:       (x1, y1, x2, y2) YOLO person bounding box.
                       Pass this for best performance — only this crop is sent
                       to DeepFace instead of the full frame.

        Returns:
            list[RecognitionResult]
        """
        if not FACE_LIB_OK or not self.encodings_loaded:
            return []

        # ── Frame skip: return cached results on skipped frames ───────
        if frame_num % max(1, self.skip) != 0:
            self._prune_tracks()
            return self._last_results

        fh, fw = frame.shape[:2]

        # ── Crop to ROI with padding ──────────────────────────────────
        if roi:
            x1, y1, x2, y2 = roi
            pad = 30                     # extra pixels around bounding box
            x1c = max(0,  x1 - pad)
            y1c = max(0,  y1 - pad)
            x2c = min(fw, x2 + pad)
            y2c = min(fh, y2 + pad)
            crop   = frame[y1c:y2c, x1c:x2c]
            offset = (x1c, y1c)
        else:
            crop   = frame
            offset = (0, 0)

        # ── Skip if crop is too small ─────────────────────────────────
        if crop.shape[0] < MIN_FACE_SIZE or crop.shape[1] < MIN_FACE_SIZE:
            self._last_results = []
            return []

        # ── Run DeepFace ──────────────────────────────────────────────
        name, confidence = self._recognize_crop(crop)
        authorized       = (name != UNKNOWN_LABEL)

        # Build bbox from crop boundaries (mapped back to full frame)
        bbox = (
            offset[0],
            offset[1],
            offset[0] + crop.shape[1],
            offset[1] + crop.shape[0],
        )

        # ── Update tracking ───────────────────────────────────────────
        track_id = self._update_tracks(name, confidence, bbox)

        result = RecognitionResult(
            name=name,
            confidence=confidence,
            authorized=authorized,
            bbox=bbox,
            track_id=track_id,
        )

        self._last_results = [result]
        return [result]

    # ──────────────────────────────────────────────────
    # FACE TRACKING  (identical to old version)
    # ──────────────────────────────────────────────────
    def _update_tracks(self, name: str, confidence: float, bbox: tuple) -> int:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

        best_tid  = None
        best_dist = 9999

        for tid, track in self._tracks.items():
            tcx = (track.bbox[0] + track.bbox[2]) // 2
            tcy = (track.bbox[1] + track.bbox[3]) // 2
            d   = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if d < best_dist and d < 150:
                best_dist = d
                best_tid  = tid

        if best_tid is not None:
            self._tracks[best_tid].update(name, confidence, bbox)
            return best_tid

        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = FaceTrack(
            track_id=tid, name=name, confidence=confidence,
            bbox=bbox, authorized=(name != UNKNOWN_LABEL)
        )
        return tid

    def _prune_tracks(self):
        now     = time.time()
        expired = [tid for tid, t in self._tracks.items()
                   if now - t.last_seen > TRACK_TIMEOUT]
        for tid in expired:
            del self._tracks[tid]

    def active_tracks(self) -> list[FaceTrack]:
        self._prune_tracks()
        return list(self._tracks.values())


# ─────────────────────────────────────────────────────────
# DRAW FACE ANNOTATIONS  (identical visual style)
# ─────────────────────────────────────────────────────────
def draw_face_results(frame: np.ndarray,
                      results: list[RecognitionResult]) -> np.ndarray:
    """
    Annotate frame with face recognition results in-place.
    AUTHORIZED  → green box + name
    UNKNOWN     → red box + INTRUDER label
    """
    for r in results:
        x1, y1, x2, y2 = r.bbox

        if r.authorized:
            color       = (0, 220, 60)
            status_text = f"AUTH: {r.name.upper()}"
        else:
            color       = (0, 30, 255)
            status_text = "INTRUDER"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner brackets — tactical look
        tl = 16
        for sx, sy, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (sx, sy), (sx + dx * tl, sy), color, 3)
            cv2.line(frame, (sx, sy), (sx, sy + dy * tl), color, 3)

        # Name label
        label_txt = f"{status_text}  {r.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label_txt, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        if r.track_id >= 0:
            cv2.putText(frame, f"ID:{r.track_id}",
                        (x2 - 42, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    return frame


# ─────────────────────────────────────────────────────────
# STATUS DICT  (same keys as old version + new DeepFace keys)
# ─────────────────────────────────────────────────────────
def face_module_status(fr: FaceRecognizer) -> dict:
    return {
        "library_ok":       FACE_LIB_OK,
        "encodings_loaded": fr.encodings_loaded,
        "known_count":      fr.encoding_count(),
        "known_people":     fr.known_people(),
        "active_tracks":    len(fr.active_tracks()),
        # DeepFace-specific info for sidebar display
        "df_model":         fr.model_name,
        "df_metric":        fr.metric,
        "df_detector":      fr.detector,
    }