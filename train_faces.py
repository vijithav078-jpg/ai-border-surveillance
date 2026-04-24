"""
train_faces.py
══════════════════════════════════════════════════════════════════════════
Face Encoding Generator for AI Border Surveillance System

Run this ONCE before starting surveillance to build the face database.

Usage:
    python train_faces.py
    python train_faces.py --dataset dataset --output encodings.pkl --show

Dataset structure:
    dataset/
    ├── person_name_1/
    │   ├── photo1.jpg
    │   ├── photo2.jpg
    │   └── photo3.jpg        ← minimum 3 images per person recommended
    ├── person_name_2/
    │   └── ...
    └── ...

Tips for best accuracy:
    • Use 5–10 photos per person
    • Include different angles (front, slight left/right)
    • Include different lighting conditions
    • Images should be at least 200×200 px
    • Face should be clearly visible (no heavy obstructions)
══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import pickle
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import face_recognition
except ImportError:
    print("\n[ERROR] face_recognition library not installed.")
    print("Install with:  pip install face_recognition")
    print("On Windows, also install cmake and dlib first:")
    print("  pip install cmake dlib face_recognition")
    sys.exit(1)


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
SUPPORTED_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET = "dataset"
DEFAULT_OUTPUT  = "encodings.pkl"
DETECTION_MODEL = "hog"   # "hog" (fast/CPU) or "cnn" (accurate/GPU)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def _print_header():
    print("\n" + "═" * 60)
    print("  AI Border Surveillance — Face Encoder")
    print("═" * 60)


def _count_images(dataset_dir: str) -> dict:
    """Returns {person_name: image_count} for all subdirectories."""
    counts = {}
    if not os.path.isdir(dataset_dir):
        return counts
    for person in sorted(os.listdir(dataset_dir)):
        person_path = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_path):
            continue
        imgs = [
            f for f in os.listdir(person_path)
            if Path(f).suffix.lower() in SUPPORTED_EXT
        ]
        counts[person] = len(imgs)
    return counts


def _load_and_validate_image(img_path: str) -> np.ndarray | None:
    """Load image and validate it is usable (non-empty, correct format)."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    if img.shape[0] < 50 or img.shape[1] < 50:
        return None   # too small to detect a face
    return img


def _encode_image(img_bgr: np.ndarray,
                  model: str = DETECTION_MODEL) -> list[np.ndarray]:
    """
    Detect and encode all faces in one image.
    Returns list of 128-d encoding vectors (one per face found).
    """
    rgb       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model=model)
    if not locations:
        return []
    encodings = face_recognition.face_encodings(rgb, locations)
    return encodings


# ─────────────────────────────────────────────────────────
# MAIN ENCODING PIPELINE
# ─────────────────────────────────────────────────────────
def build_encodings(dataset_dir: str = DEFAULT_DATASET,
                    output_path: str  = DEFAULT_OUTPUT,
                    model: str        = DETECTION_MODEL,
                    show_preview: bool = False) -> dict:
    """
    Walk dataset directory, encode all faces, save to .pkl file.

    Returns:
        {"encodings": [...], "names": [...], "stats": {...}}
    """
    _print_header()

    if not os.path.isdir(dataset_dir):
        print(f"\n[ERROR] Dataset directory not found: {dataset_dir}")
        print("Create it with the structure shown in the docstring above.")
        sys.exit(1)

    counts = _count_images(dataset_dir)
    if not counts:
        print(f"\n[ERROR] No person subdirectories found in: {dataset_dir}")
        sys.exit(1)

    print(f"\nDataset : {os.path.abspath(dataset_dir)}")
    print(f"Output  : {os.path.abspath(output_path)}")
    print(f"Model   : {model}")
    print(f"\nPersons found ({len(counts)}):")
    for person, count in counts.items():
        warn = "  ← consider adding more images" if count < 3 else ""
        print(f"  {person:30s}  {count} image(s){warn}")

    print("\nEncoding faces...")
    print("─" * 50)

    all_encodings : list[np.ndarray] = []
    all_names     : list[str]        = []
    stats = {"processed": 0, "faces_found": 0, "skipped": 0, "errors": 0}

    t_start = time.time()

    for person_name in sorted(counts.keys()):
        person_dir = os.path.join(dataset_dir, person_name)
        img_files  = sorted([
            f for f in os.listdir(person_dir)
            if Path(f).suffix.lower() in SUPPORTED_EXT
        ])

        person_count = 0
        print(f"\n  [{person_name}]")

        for img_file in img_files:
            img_path = os.path.join(person_dir, img_file)
            stats["processed"] += 1

            img = _load_and_validate_image(img_path)
            if img is None:
                print(f"    ✗  {img_file:<30}  (could not load or too small)")
                stats["errors"] += 1
                continue

            try:
                encodings = _encode_image(img, model=model)
            except Exception as e:
                print(f"    ✗  {img_file:<30}  (encoding error: {e})")
                stats["errors"] += 1
                continue

            if not encodings:
                print(f"    –  {img_file:<30}  (no face detected — skipping)")
                stats["skipped"] += 1
                continue

            if len(encodings) > 1:
                print(f"    !  {img_file:<30}  ({len(encodings)} faces — using first only)")

            # Use only the first (most prominent) face per image
            all_encodings.append(encodings[0])
            all_names.append(person_name)
            stats["faces_found"] += 1
            person_count += 1
            print(f"    ✓  {img_file:<30}  encoded")

            # Optional live preview
            if show_preview:
                rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model=model)
                for top, right, bottom, left in locs:
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 220, 60), 2)
                cv2.putText(img, person_name, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 60), 2)
                cv2.imshow(f"Preview — {person_name}", img)
                cv2.waitKey(400)

        if person_count == 0:
            print(f"    ⚠  WARNING: No usable face images for {person_name}!")

    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start

    # ── Save encodings ──
    if not all_encodings:
        print("\n[ERROR] No faces were successfully encoded. Check your dataset.")
        sys.exit(1)

    data = {"encodings": all_encodings, "names": all_names}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    # ── Summary ──
    unique_people = sorted(set(all_names))
    print("\n" + "═" * 60)
    print("  ENCODING COMPLETE")
    print("─" * 60)
    print(f"  Images processed : {stats['processed']}")
    print(f"  Faces encoded    : {stats['faces_found']}")
    print(f"  Skipped (no face): {stats['skipped']}")
    print(f"  Errors           : {stats['errors']}")
    print(f"  People in DB     : {len(unique_people)}")
    print(f"  Time taken       : {elapsed:.1f}s")
    print(f"\n  Saved to: {os.path.abspath(output_path)}")
    print("\n  Authorized persons:")
    for p in unique_people:
        n = all_names.count(p)
        print(f"    • {p}  ({n} encoding{'s' if n>1 else ''})")
    print("═" * 60)

    return {"encodings": all_encodings, "names": all_names, "stats": stats}


# ─────────────────────────────────────────────────────────
# VERIFY ENCODINGS  (quick integrity check)
# ─────────────────────────────────────────────────────────
def verify_encodings(path: str = DEFAULT_OUTPUT):
    """Load and print a summary of an existing encodings file."""
    if not os.path.exists(path):
        print(f"[verify] File not found: {path}")
        return

    with open(path, "rb") as f:
        data = pickle.load(f)

    encodings = data.get("encodings", [])
    names     = data.get("names",     [])

    if not encodings:
        print("[verify] Encodings file is empty.")
        return

    unique = sorted(set(names))
    print(f"\n[verify] {path}")
    print(f"  Total encodings : {len(encodings)}")
    print(f"  Unique persons  : {len(unique)}")
    for p in unique:
        print(f"    • {p}  ({names.count(p)} encodings)")


# ─────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build face encodings database for AI Border Surveillance"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"Path to dataset folder (default: {DEFAULT_DATASET})")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT,
                        help=f"Output .pkl path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model",   default=DETECTION_MODEL,
                        choices=["hog", "cnn"],
                        help="Detection model: hog=fast, cnn=accurate (default: hog)")
    parser.add_argument("--show",    action="store_true",
                        help="Show live preview of detected faces during encoding")
    parser.add_argument("--verify",  action="store_true",
                        help="Only verify an existing encodings file, don't re-encode")

    args = parser.parse_args()

    if args.verify:
        verify_encodings(args.output)
    else:
        build_encodings(
            dataset_dir  = args.dataset,
            output_path  = args.output,
            model        = args.model,
            show_preview = args.show,
        )


if __name__ == "__main__":
    main()
