"""
utils.py
══════════════════════════════════════════════════════════════════════════
Image Enhancement Utilities for AI Border Surveillance System
Includes: Night Vision, CLAHE, Gamma, Noise reduction, Brightness/Contrast
══════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────
# ENHANCEMENT CONFIGURATION
# ─────────────────────────────────────────────────────────
@dataclass
class NightVisionConfig:
    """All tunable parameters for the night-vision pipeline."""
    # CLAHE
    clahe_clip:       float = 3.0     # higher = more aggressive contrast boost
    clahe_grid:       int   = 8       # tile grid size (8×8)

    # Gamma correction  (< 1.0 = brighten, > 1.0 = darken)
    gamma:            float = 0.55

    # Noise reduction
    denoise:          bool  = True
    denoise_strength: int   = 7       # bilateral filter diameter

    # Brightness / contrast
    brightness:       int   = 25      # additive, 0–100
    contrast:         float = 1.25    # multiplicative, 1.0 = no change

    # Green tint (classic NV goggles look) — set to False for clean output
    green_tint:       bool  = False
    green_intensity:  float = 0.30    # 0 = none, 1 = full green channel boost

    # Sharpening
    sharpen:          bool  = True
    sharpen_amount:   float = 0.6     # 0 = off, 1 = full sharpen


# ─────────────────────────────────────────────────────────
# GAMMA CORRECTION LUT  (built once, reused every frame)
# ─────────────────────────────────────────────────────────
def _build_gamma_lut(gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return table


# ─────────────────────────────────────────────────────────
# CORE ENHANCEMENT FUNCTIONS
# ─────────────────────────────────────────────────────────
def apply_clahe(frame: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    """
    CLAHE on the Luminance channel (LAB color space).
    Boosts local contrast without over-brightening highlights.
    """
    lab            = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe          = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l_eq           = clahe.apply(l_ch)
    lab_eq         = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_gamma(frame: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    """Gamma correction using a pre-built LUT (very fast)."""
    lut = _build_gamma_lut(gamma)
    return cv2.LUT(frame, lut)


def apply_denoise(frame: np.ndarray, strength: int = 7) -> np.ndarray:
    """
    Bilateral filter: reduces noise while preserving edges.
    Slightly slower than Gaussian but far better quality for surveillance.
    """
    return cv2.bilateralFilter(frame, strength, strength * 2, strength * 2)


def apply_brightness_contrast(frame: np.ndarray,
                               brightness: int   = 25,
                               contrast:   float = 1.25) -> np.ndarray:
    """
    Adjust brightness (additive) and contrast (multiplicative).
    Uses cv2.convertScaleAbs for clipping at [0, 255].
    """
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


def apply_sharpen(frame: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """
    Unsharp mask sharpening — enhances edge detail.
    Useful for reading licence plates / faces in low-light.
    """
    blur     = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)
    return sharpened


def apply_green_tint(frame: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """
    Adds classic night-vision green phosphor tint.
    Splits channels, boosts green, recombines.
    """
    b, g, r = cv2.split(frame.astype(np.float32))
    g       = np.clip(g * (1.0 + intensity), 0, 255)
    r       = np.clip(r * (1.0 - intensity * 0.5), 0, 255)
    b       = np.clip(b * (1.0 - intensity * 0.5), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────
def enhance_night_vision(frame: np.ndarray,
                          cfg: NightVisionConfig = None,
                          night_mode: bool = True) -> np.ndarray:
    """
    Full night-vision enhancement pipeline.

    Args:
        frame:      BGR frame from OpenCV
        cfg:        NightVisionConfig — uses sensible defaults if None
        night_mode: if False, returns frame unchanged (toggle support)

    Returns:
        Enhanced BGR frame (same shape as input)

    Pipeline order:
        1. Denoise          (bilateral filter)
        2. CLAHE            (adaptive histogram eq on L channel)
        3. Gamma correction (brighten dark regions)
        4. Brightness/contrast adjustment
        5. Sharpening       (unsharp mask)
        6. Green tint       (optional — NV goggle aesthetic)
    """
    if not night_mode:
        return frame

    if cfg is None:
        cfg = NightVisionConfig()

    out = frame.copy()

    # Step 1 — Noise reduction first (cleaner input for all subsequent steps)
    if cfg.denoise:
        out = apply_denoise(out, cfg.denoise_strength)

    # Step 2 — CLAHE: boost local contrast
    out = apply_clahe(out, cfg.clahe_clip, cfg.clahe_grid)

    # Step 3 — Gamma correction: lift dark areas
    out = apply_gamma(out, cfg.gamma)

    # Step 4 — Brightness + contrast
    out = apply_brightness_contrast(out, cfg.brightness, cfg.contrast)

    # Step 5 — Sharpen: recover edge detail
    if cfg.sharpen:
        out = apply_sharpen(out, cfg.sharpen_amount)

    # Step 6 — Optional green tint
    if cfg.green_tint:
        out = apply_green_tint(out, cfg.green_intensity)

    return out


def enhance_basic(frame: np.ndarray) -> np.ndarray:
    """
    Lightweight CLAHE-only enhancement (original behaviour, kept for compatibility).
    Used when night_mode is off but low-light enhancement is still desired.
    """
    lab            = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe          = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l_ch), a_ch, b_ch]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────
# HUD OVERLAY HELPERS
# ─────────────────────────────────────────────────────────
def draw_night_vision_indicator(frame: np.ndarray, active: bool) -> np.ndarray:
    """Draw a small NV indicator badge in the top-right corner."""
    fw = frame.shape[1]
    if active:
        label = "NV ON"
        color = (0, 220, 80)
    else:
        label = "NV OFF"
        color = (80, 80, 80)
    cv2.putText(frame, label, (fw - 85, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
    cv2.circle(frame, (fw - 95, 14), 5, color, -1)
    return frame


def draw_face_rec_indicator(frame: np.ndarray, active: bool, n_known: int) -> np.ndarray:
    """Draw face-recognition status indicator."""
    fw = frame.shape[1]
    if active:
        label = f"FR ON  [{n_known} known]"
        color = (0, 180, 255)
    else:
        label = "FR OFF"
        color = (80, 80, 80)
    cv2.putText(frame, label, (fw - 200, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
    return frame


# ─────────────────────────────────────────────────────────
# ZONE + BOX DRAWING  (kept here for utils convenience)
# ─────────────────────────────────────────────────────────
def draw_zone_overlay(frame: np.ndarray, zones: list) -> None:
    """Draw restricted zone overlays on frame (in-place)."""
    for z in zones:
        ov = frame.copy()
        cv2.rectangle(ov, (z["left"], z["top"]), (z["right"], z["bottom"]), (0, 0, 200), -1)
        cv2.addWeighted(ov, 0.07, frame, 0.93, 0, frame)
        cv2.rectangle(frame, (z["left"], z["top"]), (z["right"], z["bottom"]), (0, 0, 220), 2)
        tl = 18
        for sx, sy, dx, dy in [
            (z["left"],  z["top"],    1,  1),
            (z["right"], z["top"],   -1,  1),
            (z["left"],  z["bottom"], 1, -1),
            (z["right"], z["bottom"],-1, -1),
        ]:
            cv2.line(frame, (sx, sy), (sx + dx * tl, sy), (0, 0, 255), 2)
            cv2.line(frame, (sx, sy), (sx, sy + dy * tl), (0, 0, 255), 2)
        cv2.putText(frame, z["name"], (z["left"] + 8, z["top"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)


def draw_detection(frame: np.ndarray,
                   x1: int, y1: int, x2: int, y2: int,
                   label: str, conf: float,
                   color: tuple, intruding: bool) -> None:
    """Draw YOLO detection box with label and confidence bar (in-place)."""
    lw = 3 if intruding else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, lw)
    if intruding:
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 1)
    txt = f"{label}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
    bw = int((x2 - x1) * conf)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + bw, y2 + 6), color, -1)


def draw_hud(frame: np.ndarray, fps: float, n_det: int,
             model_name: str, night_on: bool, fr_on: bool) -> None:
    """Draw heads-up display info on frame (in-place)."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    fw = frame.shape[1]
    lines = [
        f"FPS  {fps:4.1f}",
        f"DET  {n_det}",
        f"MDL  {model_name}",
        f"NV   {'ON' if night_on else 'OFF'}",
        f"FR   {'ON' if fr_on else 'OFF'}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
    cv2.putText(frame, ts, (fw - 235, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 100), 1)
