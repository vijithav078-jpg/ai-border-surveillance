import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import time
import threading
import numpy as np
import platform
from collections import deque
import sys

# ─────────────────── PAGE CONFIG ───────────────────
st.set_page_config(
    page_title="AI Border Surveillance",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ─────────────────── CROSS-PLATFORM ALARM ───────────────────
def play_alarm(severity="MEDIUM"):
    try:
        if platform.system() == "Windows":
            import winsound
            patterns = {
                "HIGH":   [(1500,150),(900,150),(1500,150),(900,150),(1500,150),(900,150)],
                "MEDIUM": [(1200,250),(1200,250),(1200,250)],
                "LOW":    [(800,300),(800,300)],
            }
            for freq, dur in patterns.get(severity, patterns["MEDIUM"]):
                winsound.Beep(freq, dur)
                time.sleep(0.08)
        else:
            os.system('echo -e "\\a"')
    except Exception:
        pass

# ─────────────────── THEME DEFINITIONS ───────────────────
THEMES = {
    "🟢 Tactical Green HUD": {
        # Radar-green on near-black — real army HUD / radar room aesthetic
        "bg":         "#020d02",
        "bg2":        "#041404",
        "bg3":        "#071a07",
        "grid":       "rgba(0,255,80,0.055)",
        "accent":     "#00e566",
        "accent2":    "#33ff88",
        "dim":        "rgba(0,229,102,0.50)",
        "red":        "#ff3b3b",
        "orange":     "#ff8800",
        "yellow":     "#ffe033",
        "blue":       "#00d4ff",
        "card":       "rgba(0,229,102,0.045)",
        "border":     "rgba(0,229,102,0.20)",
        "sidebar":    "#010801",
        "sidebar_txt":"#00e566",
        "h_shadow":   "0 0 20px rgba(0,229,102,0.55), 0 0 40px rgba(0,229,102,0.18)",
        "font_body":  "'Rajdhani', sans-serif",
        "font_head":  "'Orbitron', monospace",
        "font_mono":  "'Share Tech Mono', monospace",
        "gfont":      "Orbitron:wght@400;700;900|Rajdhani:wght@400;600;700|Share+Tech+Mono",
        "badge_txt":  "#020d02",
        "light":      False,
        "scan_line":  True,
        "glow_btn":   "rgba(0,229,102,0.35)",
    },
    "⬛ Stealth Black Ops": {
        # Pure black + blood-red accents — classified / special forces terminal
        "bg":         "#080808",
        "bg2":        "#0f0f0f",
        "bg3":        "#141414",
        "grid":       "rgba(220,30,30,0.04)",
        "accent":     "#cc2200",
        "accent2":    "#ff4422",
        "dim":        "rgba(204,34,0,0.55)",
        "red":        "#ff2200",
        "orange":     "#ff6600",
        "yellow":     "#ffaa00",
        "blue":       "#3388ff",
        "card":       "rgba(204,34,0,0.05)",
        "border":     "rgba(204,34,0,0.22)",
        "sidebar":    "#040404",
        "sidebar_txt":"#cc2200",
        "h_shadow":   "0 0 16px rgba(204,34,0,0.6), 0 0 32px rgba(204,34,0,0.2)",
        "font_body":  "'Rajdhani', sans-serif",
        "font_head":  "'Oswald', sans-serif",
        "font_mono":  "'Share Tech Mono', monospace",
        "gfont":      "Oswald:wght@400;500;600;700|Rajdhani:wght@400;600;700|Share+Tech+Mono",
        "badge_txt":  "#080808",
        "light":      False,
        "scan_line":  False,
        "glow_btn":   "rgba(204,34,0,0.40)",
    },
    "🌊 Naval Command Center": {
        # Deep navy + electric cyan — defense control room / fleet ops
        "bg":         "#060e1a",
        "bg2":        "#0a1628",
        "bg3":        "#0e1f36",
        "grid":       "rgba(0,180,255,0.05)",
        "accent":     "#00b4ff",
        "accent2":    "#33ccff",
        "dim":        "rgba(0,180,255,0.50)",
        "red":        "#ff4055",
        "orange":     "#ff8c22",
        "yellow":     "#ffd700",
        "blue":       "#00b4ff",
        "card":       "rgba(0,180,255,0.05)",
        "border":     "rgba(0,180,255,0.20)",
        "sidebar":    "#030a12",
        "sidebar_txt":"#00b4ff",
        "h_shadow":   "0 0 18px rgba(0,180,255,0.55), 0 0 36px rgba(0,180,255,0.15)",
        "font_body":  "'Inter', sans-serif",
        "font_head":  "'Exo 2', monospace",
        "font_mono":  "'JetBrains Mono', monospace",
        "gfont":      "Exo+2:wght@600;700;800|Inter:wght@400;500;600|JetBrains+Mono",
        "badge_txt":  "#060e1a",
        "light":      False,
        "scan_line":  False,
        "glow_btn":   "rgba(0,180,255,0.35)",
    },
    "🏜️ Desert Tactical": {
        # Sand / khaki + amber — border patrol, desert ops, India terrain realism
        "bg":         "#0f0c08",
        "bg2":        "#16110a",
        "bg3":        "#1e1710",
        "grid":       "rgba(210,165,80,0.05)",
        "accent":     "#d4a044",
        "accent2":    "#f0c060",
        "dim":        "rgba(212,160,68,0.52)",
        "red":        "#e03030",
        "orange":     "#e07020",
        "yellow":     "#f0c060",
        "blue":       "#6090c8",
        "card":       "rgba(212,160,68,0.055)",
        "border":     "rgba(212,160,68,0.22)",
        "sidebar":    "#0a0804",
        "sidebar_txt":"#d4a044",
        "h_shadow":   "0 0 16px rgba(212,160,68,0.45)",
        "font_body":  "'Rajdhani', sans-serif",
        "font_head":  "'Oswald', sans-serif",
        "font_mono":  "'Share Tech Mono', monospace",
        "gfont":      "Oswald:wght@400;500;600;700|Rajdhani:wght@400;600;700|Share+Tech+Mono",
        "badge_txt":  "#0f0c08",
        "light":      False,
        "scan_line":  False,
        "glow_btn":   "rgba(212,160,68,0.35)",
    },
    # ── Classic / Legacy themes ──────────────────────────────────────────
    "⬜ White & Black": {
        # Clean white background, jet-black text — sharp, minimal, professional
        # Sidebar stays dark (black) for contrast
        "bg":         "#f5f6f8",
        "bg2":        "#ffffff",
        "bg3":        "#eceef2",
        "grid":       "rgba(0,0,0,0.045)",
        "accent":     "#111111",
        "accent2":    "#333333",
        "dim":        "rgba(0,0,0,0.40)",
        "red":        "#cc1111",
        "orange":     "#c85a00",
        "yellow":     "#a07000",
        "blue":       "#1155cc",
        "card":       "rgba(0,0,0,0.04)",
        "border":     "rgba(0,0,0,0.13)",
        "sidebar":    "#111111",
        "sidebar_txt":"#eeeeee",
        "h_shadow":   "none",
        "font_body":  "'Inter', sans-serif",
        "font_head":  "'Exo 2', 'Inter', sans-serif",
        "font_mono":  "'JetBrains Mono', monospace",
        "gfont":      "Exo+2:wght@600;700;800|Inter:wght@400;500;600|JetBrains+Mono",
        "badge_txt":  "#ffffff",
        "light":      True,
        "scan_line":  False,
        "glow_btn":   "rgba(0,0,0,0.15)",
    },
    "🪖 Military Black": {
        # Near-black background, bright white/light-grey text
        # High-contrast tactical field aesthetic — Oswald bold condensed headers
        "bg":         "#0e0e0e",
        "bg2":        "#141414",
        "bg3":        "#1c1c1c",
        "grid":       "rgba(255,255,255,0.025)",
        "accent":     "#e2e2e2",
        "accent2":    "#ffffff",
        "dim":        "rgba(226,226,226,0.50)",
        "red":        "#ff4444",
        "orange":     "#ff8c00",
        "yellow":     "#ffd700",
        "blue":       "#4da6ff",
        "card":       "rgba(255,255,255,0.03)",
        "border":     "rgba(255,255,255,0.11)",
        "sidebar":    "#080808",
        "sidebar_txt":"#e2e2e2",
        "h_shadow":   "0 0 12px rgba(226,226,226,0.22)",
        "font_body":  "'Rajdhani', sans-serif",
        "font_head":  "'Oswald', sans-serif",
        "font_mono":  "'Share Tech Mono', monospace",
        "gfont":      "Oswald:wght@400;500;600;700|Rajdhani:wght@400;600;700|Share+Tech+Mono",
        "badge_txt":  "#0e0e0e",
        "light":      False,
        "scan_line":  False,
        "glow_btn":   "rgba(226,226,226,0.22)",
    },
    "⚓ Navy & Gold": {
        # Deep navy background, warm gold accents — command authority look
        # Cinzel serif headers give a formal military-command aesthetic
        "bg":         "#07101e",
        "bg2":        "#0b1828",
        "bg3":        "#0f2035",
        "grid":       "rgba(197,160,56,0.055)",
        "accent":     "#c5a038",
        "accent2":    "#e8c060",
        "dim":        "rgba(197,160,56,0.55)",
        "red":        "#e84040",
        "orange":     "#e07830",
        "yellow":     "#e8c060",
        "blue":       "#4090d8",
        "card":       "rgba(197,160,56,0.06)",
        "border":     "rgba(197,160,56,0.22)",
        "sidebar":    "#040c16",
        "sidebar_txt":"#c5a038",
        "h_shadow":   "0 0 18px rgba(197,160,56,0.50)",
        "font_body":  "'Rajdhani', sans-serif",
        "font_head":  "'Cinzel', serif",
        "font_mono":  "'Share Tech Mono', monospace",
        "gfont":      "Cinzel:wght@400;600;700|Rajdhani:wght@400;600;700|Share+Tech+Mono",
        "badge_txt":  "#07101e",
        "light":      False,
        "scan_line":  False,
        "glow_btn":   "rgba(197,160,56,0.35)",
    },
}

# ─────────────────── THEME SELECTION (early, before sidebar) ───────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "🟢 Tactical Green HUD"

# ─────────────────── APPLY THEME ───────────────────
def apply_theme(t):
    T = THEMES[t]
    is_light = T.get("light", False)
    # For light theme, input text/widget text must be dark
    input_color  = "#111111" if is_light else "var(--accent)"
    input_bg     = "#ffffff" if is_light else "var(--bg2)"
    main_color   = "#111111" if is_light else "var(--accent)"
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family={T["gfont"]}&display=swap');

:root {{
    --bg:          {T["bg"]};
    --bg2:         {T["bg2"]};
    --bg3:         {T["bg3"]};
    --grid:        {T["grid"]};
    --accent:      {T["accent"]};
    --accent2:     {T["accent2"]};
    --dim:         {T["dim"]};
    --red:         {T["red"]};
    --orange:      {T["orange"]};
    --yellow:      {T["yellow"]};
    --blue:        {T["blue"]};
    --card:        {T["card"]};
    --border:      {T["border"]};
    --sidebar:     {T["sidebar"]};
    --sidebar-txt: {T["sidebar_txt"]};
    --h-shadow:    {T["h_shadow"]};
    --font-body:   {T["font_body"]};
    --font-head:   {T["font_head"]};
    --font-mono:   {T["font_mono"]};
    --badge-txt:   {T["badge_txt"]};
    --main-color:  {main_color};
}}

html, body, .stApp {{
    background-color: var(--bg) !important;
    background-image:
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px) !important;
    background-size: 44px 44px !important;
    font-family: var(--font-body);
    color: var(--main-color);
}}

/* light-theme: make all default streamlit text dark */
{"p, span, div, li, label { color: #111111 !important; }" if is_light else ""}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--sidebar-txt) !important; }}
[data-testid="stSidebar"] .stRadio > label {{
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
}}

/* ── Headings ── */
h1, h2, h3 {{
    font-family: var(--font-head) !important;
    color: var(--accent) !important;
    text-shadow: var(--h-shadow);
}}
h1 {{ font-size: 1.6rem !important; letter-spacing: 3px; }}
h2 {{ font-size: 1.15rem !important; letter-spacing: 2px; }}
h3 {{ font-size: 0.95rem !important; letter-spacing: 1.5px; }}

/* ── Buttons ── */
.stButton > button {{
    background: var(--bg2) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 4px !important;
    font-family: var(--font-head) !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px;
    padding: 7px 16px !important;
    transition: all 0.22s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    text-transform: uppercase;
}}
.stButton > button:hover {{
    box-shadow: 0 0 22px {T["glow_btn"]}, 0 0 6px {T["glow_btn"]} !important;
    background: var(--card) !important;
    transform: translateY(-1px);
    border-color: var(--accent2) !important;
}}
.stButton > button:active {{
    transform: translateY(0px);
    box-shadow: 0 0 10px {T["glow_btn"]} !important;
}}

/* ── Form labels ── */
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stMultiSelect"] label {{
    color: var(--accent2) !important;
    font-family: var(--font-body) !important;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    background: var(--bg2) !important;
}}
hr {{ border-color: var(--border) !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: var(--bg2); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}

/* ────────────────── KPI GRID ────────────────── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 16px;
}}
.kpi-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 14px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(4px);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.5);
}}
.kpi-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.8;
}}
.kpi-card::after {{
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}}
.kpi-val {{
    font-family: var(--font-head);
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--accent);
    text-shadow: var(--h-shadow);
    line-height: 1.1;
    letter-spacing: 1px;
}}
.kpi-val.danger {{ color: var(--red);    text-shadow: 0 0 14px var(--red); }}
.kpi-val.warn   {{ color: var(--yellow); text-shadow: 0 0 14px var(--yellow); }}
.kpi-val.info   {{ color: var(--blue);   text-shadow: 0 0 14px var(--blue); }}
.kpi-lbl {{
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--dim);
    letter-spacing: 2.5px;
    margin-top: 7px;
    text-transform: uppercase;
}}

/* ────────────────── ALERT BOXES ────────────────── */
.alert-box {{
    background: rgba(240,82,82,0.09);
    border: 1px solid var(--red);
    border-left: 4px solid var(--red);
    border-radius: 8px;
    padding: 11px 16px;
    margin: 6px 0;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--red);
    text-shadow: 0 0 8px var(--red);
}}
.alert-box.warn {{
    background: rgba(251,191,36,0.07);
    border-color: var(--yellow);
    border-left-color: var(--yellow);
    color: var(--yellow);
    text-shadow: 0 0 8px var(--yellow);
}}
.alert-box.ok {{
    background: var(--card);
    border-color: var(--accent);
    border-left-color: var(--accent);
    color: var(--accent);
    text-shadow: 0 0 6px var(--accent);
}}
.alert-box.info {{
    background: rgba(56,189,248,0.07);
    border-color: var(--blue);
    border-left-color: var(--blue);
    color: var(--blue);
    text-shadow: 0 0 6px var(--blue);
}}

/* ────────────────── LIVE LOG ────────────────── */
.log-entry {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.73rem;
    transition: background 0.15s;
}}
.log-entry:hover {{ background: var(--card); }}
.log-entry:nth-child(odd) {{ background: rgba(0,0,0,0.12); }}

/* ────────────────── BADGES ────────────────── */
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.6rem;
    font-family: var(--font-head);
    font-weight: 700;
    letter-spacing: 1px;
}}
.badge-PERSON     {{ background:rgba(240,82,82,0.2);   border:1px solid var(--red);   color:var(--red); }}
.badge-VEHICLE    {{ background:rgba(251,146,60,0.18); border:1px solid var(--orange); color:var(--orange); }}
.badge-ANIMAL     {{ background:rgba(56,189,248,0.18); border:1px solid var(--blue);   color:var(--blue); }}
.badge-WEAPON     {{ background:rgba(255,0,100,0.22);  border:1px solid #ff0064;       color:#ff0064; }}
.badge-OBJECT     {{ background:rgba(251,191,36,0.18); border:1px solid var(--yellow); color:var(--yellow); }}
.badge-ELECTRONIC {{ background:rgba(167,139,250,0.2); border:1px solid #a78bfa;       color:#a78bfa; }}

/* ────────────────── SECTION TITLE ────────────────── */
.section-title {{
    font-family: var(--font-head);
    font-size: 0.65rem;
    letter-spacing: 3.5px;
    color: var(--dim);
    margin: 16px 0 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 5px;
    text-transform: uppercase;
}}

/* ────────────────── STANDBY ────────────────── */
.standby {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 80px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.standby::before {{
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, var(--card) 0%, transparent 65%);
}}

/* ────────────────── HEADER BANNER ────────────────── */
.header-banner {{
    background: linear-gradient(135deg, var(--bg2) 0%, var(--bg3) 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 28px 16px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}}
.header-banner::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent);
}}
.header-banner::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 30%; right: 30%; height: 1px;
    background: var(--border);
}}
.header-title {{
    font-family: var(--font-head);
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--accent);
    text-shadow: var(--h-shadow);
    letter-spacing: 4px;
    margin: 0;
}}
.header-sub {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 5px;
    margin-top: 4px;
}}
.header-status {{
    position: absolute;
    top: 18px; right: 24px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 2px;
    text-align: right;
}}
.status-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s infinite;
    margin-right: 6px;
}}
@keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:0.5; transform:scale(0.85); }}
}}

/* ────────────────── THEME PILL ────────────────── */
.theme-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    background: var(--card);
    border: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--dim);
    margin-bottom: 8px;
}}

/* ────────────────── ALARM FLASH ────────────────── */
@keyframes alarmFlash {{
    0%,100% {{ opacity:1; }}
    50%     {{ opacity:0.3; }}
}}
.alarm-flash {{ animation: alarmFlash 0.5s infinite; }}
</style>
""", unsafe_allow_html=True)

    # ── CRT scanline + radar ping — injected separately to avoid f-string brace conflicts ──
    if T.get("scan_line"):
        st.markdown("""
<style>
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.10) 2px,
        rgba(0,0,0,0.10) 4px
    );
    pointer-events: none;
    z-index: 9999;
}
@keyframes radarPing {
    0%   { box-shadow: 0 0 0 0 rgba(0,229,102,0.55); }
    70%  { box-shadow: 0 0 0 14px rgba(0,229,102,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,229,102,0); }
}
.kpi-card { animation: radarPing 2.8s infinite; }
</style>
""", unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# ─────────────────── JS AUDIO ALARM (browser-side) ───────────────────
def inject_alarm_js(severity="MEDIUM"):
    """Inject Web Audio API alarm that plays in the browser immediately."""
    patterns = {
        "HIGH":   [(1500,0.12),(900,0.10),(1500,0.12),(900,0.10),(1500,0.12),(900,0.10)],
        "MEDIUM": [(1200,0.18),(1200,0.18),(1200,0.18)],
        "LOW":    [(800,0.22),(800,0.22)],
    }
    beeps = patterns.get(severity, patterns["MEDIUM"])
    beep_js = ", ".join(f"[{f},{d}]" for f,d in beeps)
    st.markdown(f"""
<script>
(function(){{
  const beeps = [{beep_js}];
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  let t = ctx.currentTime;
  beeps.forEach(([freq, dur]) => {{
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = freq;
    osc.type = 'square';
    gain.gain.setValueAtTime(0.35, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);
    osc.start(t); osc.stop(t + dur + 0.01);
    t += dur + 0.09;
  }});
}})();
</script>
""", unsafe_allow_html=True)

# ─────────────────── HEADER ───────────────────
now_str = datetime.now().strftime("%Y-%m-%d  %H:%M")
st.markdown(f"""
<div class='header-banner'>
  <div class='header-status'>
    <span class='status-dot'></span>SYSTEM ONLINE<br>
    <span style='font-size:0.58rem;opacity:0.6;'>{now_str}</span>
  </div>
  <div class='header-title'>🛡️ AI BORDER SURVEILLANCE</div>
  <div class='header-sub'>YOLOV8 · REAL-TIME · MULTI-CLASS DETECTION · ZONE INTRUSION ALERT</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────── PATHS & LOG ───────────────────
os.makedirs("intruders", exist_ok=True)

LOG_FILE = "intrusion_log.csv"
COLS     = ["Time", "Type", "Confidence", "Zone", "Severity", "Status"]

def ensure_log():
    """Create or repair the CSV log."""
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=COLS).to_csv(LOG_FILE, index=False)
        return
    try:
        df_check = pd.read_csv(LOG_FILE, nrows=0)
        missing  = [c for c in COLS if c not in df_check.columns]
        if missing:
            # migrate: rename old file and start fresh
            os.rename(LOG_FILE, LOG_FILE + ".bak")
            pd.DataFrame(columns=COLS).to_csv(LOG_FILE, index=False)
    except Exception:
        pd.DataFrame(columns=COLS).to_csv(LOG_FILE, index=False)

ensure_log()

def read_log():
    """Returns clean DataFrame, never raises."""
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"]).sort_values("Time", ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=COLS)

def append_log(itype, conf, zone_name, severity):
    row = {
        "Time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Type":       itype,
        "Confidence": round(float(conf), 4),
        "Zone":       zone_name,
        "Severity":   severity,
        "Status":     f"INTRUSION-{severity}",
    }
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=False, index=False)
    st.session_state.live_log.appendleft(row)
    st.session_state.total_today  += 1
    st.session_state.zone_breaches += 1

# ─────────────────── MODEL LOAD ───────────────────
@st.cache_resource
def load_model(path):
    try:
        return YOLO(path), None
    except Exception as e:
        return None, str(e)

# ─────────────────── CLASS MAP ───────────────────
CLASS_MAP = {
    "person":     ("PERSON",     (30,  30, 255), "HIGH"),
    "car":        ("VEHICLE",    (30, 140, 255), "HIGH"),
    "truck":      ("VEHICLE",    (30, 140, 255), "HIGH"),
    "bus":        ("VEHICLE",    (30, 140, 255), "HIGH"),
    "motorcycle": ("VEHICLE",    (30, 140, 255), "MEDIUM"),
    "bicycle":    ("VEHICLE",    (30, 140, 255), "LOW"),
    "boat":       ("VEHICLE",    (30, 140, 255), "HIGH"),
    "airplane":   ("VEHICLE",    (30,  80, 255), "HIGH"),
    "dog":        ("ANIMAL",     (255,200,  30), "MEDIUM"),
    "cat":        ("ANIMAL",     (255,200,  30), "LOW"),
    "cow":        ("ANIMAL",     (255,200,  30), "MEDIUM"),
    "horse":      ("ANIMAL",     (255,200,  30), "MEDIUM"),
    "sheep":      ("ANIMAL",     (255,200,  30), "LOW"),
    "bird":       ("ANIMAL",     (255,200,  30), "LOW"),
    "bear":       ("ANIMAL",     (0,   80, 255), "HIGH"),
    "elephant":   ("ANIMAL",     (0,   80, 255), "HIGH"),
    "knife":      ("WEAPON",     (0,   0,  255), "HIGH"),
    "scissors":   ("WEAPON",     (0,   0,  200), "MEDIUM"),
    "backpack":   ("OBJECT",     (0,  200, 255), "MEDIUM"),
    "suitcase":   ("OBJECT",     (0,  200, 255), "MEDIUM"),
    "handbag":    ("OBJECT",     (0,  200, 255), "LOW"),
    "laptop":     ("ELECTRONIC", (180, 30, 255), "MEDIUM"),
    "cell phone": ("ELECTRONIC", (180, 30, 255), "LOW"),
    "tv":         ("ELECTRONIC", (180, 30, 255), "LOW"),
}
SEV_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# ─────────────────── SESSION STATE ───────────────────
for k, v in {
    "run": False,
    "live_log": deque(maxlen=40),
    "total_today": 0,
    "zone_breaches": 0,
    "alarm_active": False,
    "last_snap_time": 0,
    "last_alarm_time": 0,
    "theme": "🖥️ Professional",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────── SIDEBAR ───────────────────
with st.sidebar:
    st.markdown("<div class='section-title'>NAVIGATION</div>", unsafe_allow_html=True)
    menu = st.radio("", ["🎥 Live Surveillance", "📊 Dashboard", "🗂 Evidence", "⚙️ Settings"],
                    label_visibility="collapsed")

    st.markdown("<div class='section-title'>THEME</div>", unsafe_allow_html=True)
    # Guard: if stored theme no longer exists (e.g. old session), reset to default
    _theme_keys = list(THEMES.keys())
    if st.session_state.theme not in _theme_keys:
        st.session_state.theme = "🟢 Tactical Green HUD"
    chosen_theme = st.selectbox(
        "UI Theme", _theme_keys,
        index=_theme_keys.index(st.session_state.theme),
        label_visibility="collapsed",
    )
    if chosen_theme != st.session_state.theme:
        st.session_state.theme = chosen_theme
        st.rerun()
    st.markdown(f"<div class='theme-pill'>{chosen_theme.upper()}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>MODEL</div>", unsafe_allow_html=True)
    model_choice = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        index=1,
        help="n=tiny/fast  s=balanced  m=accurate  l=most accurate/slow"
    )
    model, model_err = load_model(model_choice)

    st.markdown("<div class='section-title'>DETECTION TUNING</div>", unsafe_allow_html=True)
    conf_thresh  = st.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
    iou_thresh   = st.slider("IoU / NMS Threshold",  0.10, 0.90, 0.40, 0.05)
    smoothing    = st.slider("Temporal Smoothing (frames)", 1, 6, 3, 1,
                              help="Require detection in N consecutive frames before alerting")
    img_size     = st.select_slider("Inference Size (px)", [320,416,512,640,768,1024], value=640)
    enhance_img  = st.checkbox("Enhance Low-Light Frames", value=True)
    filter_cls   = st.multiselect("Only Detect Classes", list(CLASS_MAP.keys()))

    st.markdown("<div class='section-title'>RESTRICTED ZONE</div>", unsafe_allow_html=True)
    zone_top    = st.slider("Zone Top",    0, 720,  180, 10)
    zone_bottom = st.slider("Zone Bottom", 0, 720,  420, 10)
    zone_left   = st.slider("Zone Left",   0, 1280, 0,   10)
    zone_right  = st.slider("Zone Right",  0, 1280, 640, 10)

    st.markdown("<div class='section-title'>CAMERA & ALERTS</div>", unsafe_allow_html=True)
    cam_idx       = st.number_input("Camera Index", 0, 10, 0, 1)
    alarm_on      = st.checkbox("Enable Alarm", value=True)
    snap_cooldown = st.slider("Snapshot Cooldown (s)", 1, 30, 4)
    show_hud      = st.checkbox("Show HUD Overlay", value=True)

# ─────────────────── ZONE HELPERS ───────────────────
ZONES = [{"name":"RESTRICTED ZONE","top":zone_top,"bottom":zone_bottom,
          "left":zone_left,"right":zone_right}]

def pt_in_zone(cx, cy, z):
    return z["left"] <= cx <= z["right"] and z["top"] <= cy <= z["bottom"]

# ─────────────────── FRAME HELPERS ───────────────────
def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def draw_zone_overlay(frame, zones):
    for z in zones:
        ov = frame.copy()
        cv2.rectangle(ov, (z["left"],z["top"]), (z["right"],z["bottom"]), (0,0,200), -1)
        cv2.addWeighted(ov, 0.07, frame, 0.93, 0, frame)
        cv2.rectangle(frame, (z["left"],z["top"]), (z["right"],z["bottom"]), (0,0,220), 2)
        tl = 18
        for sx,sy,dx,dy in [(z["left"],z["top"],1,1),(z["right"],z["top"],-1,1),
                             (z["left"],z["bottom"],1,-1),(z["right"],z["bottom"],-1,-1)]:
            cv2.line(frame,(sx,sy),(sx+dx*tl,sy),(0,0,255),2)
            cv2.line(frame,(sx,sy),(sx,sy+dy*tl),(0,0,255),2)
        cv2.putText(frame, z["name"], (z["left"]+8, z["top"]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,60,255), 2)

def draw_detection(frame, x1,y1,x2,y2, label, conf, color, intruding):
    lw = 3 if intruding else 2
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,lw)
    if intruding:
        cv2.rectangle(frame,(x1-2,y1-2),(x2+2,y2+2),(0,0,255),1)
    txt = f"{label}  {conf:.0%}"
    (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+6,y1),color,-1)
    cv2.putText(frame,txt,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),2)
    bw = int((x2-x1)*conf)
    cv2.rectangle(frame,(x1,y2+2),(x1+bw,y2+6),color,-1)

def draw_hud(frame, fps, n_det, fw):
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    for i,(line) in enumerate([f"FPS  {fps:4.1f}", f"DET  {n_det}", f"MDL  {model_choice}"]):
        cv2.putText(frame,line,(10,22+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,100),1)
    cv2.putText(frame,ts,(fw-235,22),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,255,100),1)

# ─────────────────── LIVE LOG HTML ───────────────────
def render_live_log(entries):
    if not entries:
        return "<div class='alert-box ok' style='font-size:0.75rem;'>No intrusions yet.</div>"
    html = ""
    for e in list(entries)[:10]:
        t  = str(e.get("Time",""))[-8:]
        tp = e.get("Type","?")
        sv = e.get("Severity","")
        cf = e.get("Confidence",0)
        try:  cf_str = f"{float(cf):.0%}"
        except: cf_str = str(cf)
        sv_col = "#ff3333" if sv=="HIGH" else "#ffcc00" if sv=="MEDIUM" else "#00ff9f"
        html += (f"<div class='log-entry'>"
                 f"<span style='color:rgba(0,255,159,0.45)'>{t}</span>"
                 f"<span class='badge badge-{tp}'>{tp}</span>"
                 f"<span>{cf_str}</span>"
                 f"<span style='color:{sv_col}'>{sv}</span>"
                 f"</div>")
    return html

# ═══════════════════════════════════════════════════════
# PAGE: LIVE SURVEILLANCE
# ═══════════════════════════════════════════════════════
if menu == "🎥 Live Surveillance":

    if model_err:
        st.markdown(f'<div class="alert-box">MODEL ERROR: {model_err}</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 1])

    with left_col:
        b1,b2,b3 = st.columns(3)
        with b1:
            if st.button("▶  START SURVEILLANCE"):
                st.session_state.run = True
        with b2:
            if st.button("⛔  STOP"):
                st.session_state.run = False
        with b3:
            if st.button("📸  FORCE SNAPSHOT"):
                st.session_state.last_snap_time = 0
        feed_ph   = st.empty()
        status_ph = st.empty()

    with right_col:
        st.markdown("<div class='section-title'>LIVE INTEL</div>", unsafe_allow_html=True)
        kpi_ph = st.empty()
        log_ph = st.empty()

    if not st.session_state.run:
        feed_ph.markdown("""
        <div class='standby'>
          <p style='font-family:var(--font-head),Orbitron,monospace;font-size:1.1rem;color:var(--dim);letter-spacing:5px;margin:0;'>
            ◉ SYSTEM STANDBY
          </p>
          <p style='font-family:var(--font-mono),monospace;font-size:0.7rem;color:var(--dim);opacity:0.5;margin-top:12px;letter-spacing:2px;'>
            Press ▶ START SURVEILLANCE to activate camera feed
          </p>
        </div>""", unsafe_allow_html=True)

    if st.session_state.run and model:
        cap = cv2.VideoCapture(int(cam_idx))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_times = deque(maxlen=30)
        track_buf   = {}   # label -> deque of bool

        while st.session_state.run:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                status_ph.markdown('<div class="alert-box">CAMERA NOT FOUND — check Camera Index.</div>',
                                   unsafe_allow_html=True)
                time.sleep(1)
                continue

            fh, fw = frame.shape[:2]
            inf_frame = enhance_frame(frame) if enhance_img else frame
            draw_zone_overlay(frame, ZONES)

            yolo_kw = {"verbose":False,"conf":conf_thresh,"iou":iou_thresh,"imgsz":img_size}
            if filter_cls:
                ids = [i for i,n in model.names.items() if n in filter_cls]
                if ids: yolo_kw["classes"] = ids

            results = model(inf_frame, **yolo_kw)
            intrusions = []   # (itype, conf, severity, label)

            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cls   = int(box.cls[0])
                    label = model.names[cls]
                    conf  = float(box.conf[0])
                    itype,color,severity = CLASS_MAP.get(label,(label.upper(),(80,200,80),"LOW"))

                    cx,cy   = (x1+x2)//2, (y1+y2)//2
                    in_zone = any(pt_in_zone(cx,cy,z) for z in ZONES)

                    buf = track_buf.setdefault(label, deque(maxlen=smoothing))
                    buf.append(in_zone)
                    confirmed = sum(buf) >= max(1, smoothing-1)

                    draw_detection(frame,x1,y1,x2,y2,label,conf,color,intruding=confirmed)
                    if confirmed:
                        intrusions.append((itype,conf,severity,label))

            now = time.time()
            if intrusions:
                best = max(intrusions, key=lambda t: SEV_RANK.get(t[2],0))
                itype,conf,severity,_ = best
                sev_bgr = {"HIGH":(0,0,255),"MEDIUM":(0,130,255),"LOW":(0,200,80)}.get(severity,(0,0,255))
                cv2.putText(frame,f"! {itype} INTRUSION [{severity}]",
                            (20,72),cv2.FONT_HERSHEY_SIMPLEX,0.9,sev_bgr,3)
                cv2.putText(frame,f"CONF {conf:.0%}",
                            (20,102),cv2.FONT_HERSHEY_SIMPLEX,0.58,sev_bgr,2)

                # ── Alarm: fires as soon as intrusion detected, independent of snapshot ──
                alarm_cooldown_secs = 3  # re-trigger alarm at most every 3s
                if alarm_on and (now - st.session_state.last_alarm_time > alarm_cooldown_secs):
                    st.session_state.last_alarm_time = now
                    st.session_state.alarm_active = True
                    # Browser-side audio (works cross-platform in the web UI)
                    inject_alarm_js(severity)
                    # Native OS beep as fallback (background thread)
                    def _alarm(sev, s):
                        play_alarm(sev); s.alarm_active = False
                    threading.Thread(target=_alarm, args=(severity, st.session_state), daemon=True).start()

                # ── Snapshot + log (separate cooldown, not tied to alarm) ──
                if now - st.session_state.last_snap_time > snap_cooldown:
                    st.session_state.last_snap_time = now
                    fname = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"intruders/{fname}_{itype}_{severity}.jpg", frame)
                    append_log(itype, conf, "RESTRICTED ZONE", severity)

                sc = "#ff3333" if severity=="HIGH" else "#ffcc00" if severity=="MEDIUM" else "#00ff9f"
                status_ph.markdown(
                    f'<div class="alert-box">🚨 {itype} | SEVERITY: <span style="color:{sc}">{severity}</span>'
                    f' | CONF: {conf:.0%} | {datetime.now().strftime("%H:%M:%S")}</div>',
                    unsafe_allow_html=True)
            else:
                # Zone is clear — re-arm alarm for next intrusion
                st.session_state.alarm_active = False
                status_ph.markdown(
                    '<div class="alert-box ok">✔ ZONE CLEAR</div>', unsafe_allow_html=True)

            frame_times.append(time.time()-t0)
            fps = 1.0/(sum(frame_times)/len(frame_times)) if frame_times else 0
            if show_hud:
                draw_hud(frame, fps, sum(len(r.boxes) for r in results), fw)

            feed_ph.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            kpi_ph.markdown(f"""
            <div class='kpi-card' style='margin-bottom:8px;'>
              <div class='kpi-val {"danger" if st.session_state.total_today>0 else ""}'>{st.session_state.total_today}</div>
              <div class='kpi-lbl'>TODAY'S INTRUSIONS</div>
            </div>
            <div class='kpi-card' style='margin-bottom:8px;'>
              <div class='kpi-val info'>{st.session_state.zone_breaches}</div>
              <div class='kpi-lbl'>ZONE BREACHES</div>
            </div>
            <div class='kpi-card'>
              <div class='kpi-val' style='font-size:1.1rem'>{fps:.1f}</div>
              <div class='kpi-lbl'>FPS</div>
            </div>""", unsafe_allow_html=True)

            log_ph.markdown(
                "<div style='margin-top:10px;'>"
                "<div class='section-title'>RECENT EVENTS</div>"
                + render_live_log(st.session_state.live_log)
                + "</div>", unsafe_allow_html=True)

        cap.release()

# ═══════════════════════════════════════════════════════
# PAGE: DASHBOARD  (FULLY FIXED)
# ═══════════════════════════════════════════════════════
elif menu == "📊 Dashboard":

    st.markdown("## 📊 INTELLIGENCE DASHBOARD")

    # ── auto-refresh button ──
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

    df = read_log()

    if df.empty:
        st.markdown("""
        <div class='alert-box info'>
          📂 No intrusion data yet.<br>
          Start <b>Live Surveillance</b>, trigger a detection, and come back here.
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── derived columns ──
    df["Hour"]     = df["Time"].dt.hour
    df["Date"]     = df["Time"].dt.date
    # normalise Severity — handle legacy rows that stored it only in Status
    if "Severity" not in df.columns or df["Severity"].isna().all():
        df["Severity"] = df["Status"].str.extract(r"-(HIGH|MEDIUM|LOW)", expand=False)
    df["Severity"] = df["Severity"].fillna(
        df["Status"].str.extract(r"-(HIGH|MEDIUM|LOW)", expand=False)
    ).fillna("LOW")

    today      = datetime.now().date()
    df_today   = df[df["Date"] == today]
    df_yest    = df[df["Date"] == (pd.Timestamp(today) - pd.Timedelta(days=1)).date()]

    top_type   = df["Type"].mode().iloc[0]       if not df["Type"].dropna().empty   else "—"
    peak_hour  = int(df["Hour"].mode().iloc[0])  if not df["Hour"].dropna().empty   else 0
    high_count = int((df["Severity"] == "HIGH").sum())
    med_count  = int((df["Severity"] == "MEDIUM").sum())
    low_count  = int((df["Severity"] == "LOW").sum())

    # ── KPI row 1 ──
    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='kpi-card'>
        <div class='kpi-val danger'>{len(df)}</div>
        <div class='kpi-lbl'>TOTAL INTRUSIONS</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-val warn'>{len(df_today)}</div>
        <div class='kpi-lbl'>TODAY'S ALERTS</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-val' style='font-size:1.3rem'>{top_type}</div>
        <div class='kpi-lbl'>TOP THREAT TYPE</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-val info'>{peak_hour:02d}:00</div>
        <div class='kpi-lbl'>PEAK HOUR</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI row 2 ──
    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='kpi-card'>
        <div class='kpi-val danger'>{high_count}</div>
        <div class='kpi-lbl'>HIGH SEVERITY</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-val warn'>{med_count}</div>
        <div class='kpi-lbl'>MEDIUM SEVERITY</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-val'>{low_count}</div>
        <div class='kpi-lbl'>LOW SEVERITY</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-val info'>{df["Type"].nunique()}</div>
        <div class='kpi-lbl'>UNIQUE THREAT TYPES</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📈 Hourly Activity")
        hourly = df.groupby("Hour").size().reindex(range(24), fill_value=0).rename("Intrusions")
        st.area_chart(hourly, height=220)

        st.subheader("📅 Daily Trend")
        daily = df.groupby("Date").size().rename("Intrusions")
        st.line_chart(daily, height=200)

        st.subheader("📋 Recent Log")
        disp = df.head(25).copy()
        disp["Time"] = disp["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        disp["Confidence"] = pd.to_numeric(disp["Confidence"],errors="coerce").map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "")
        st.dataframe(disp[["Time","Type","Confidence","Zone","Severity"]],
                     use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("🎯 Intrusion Types")
        type_counts = df["Type"].value_counts().rename("Count")
        st.bar_chart(type_counts, height=220)

        st.subheader("⚡ Severity Distribution")
        sev_counts = (df["Severity"].value_counts()
                      .reindex(["HIGH","MEDIUM","LOW"], fill_value=0)
                      .rename("Count"))
        st.bar_chart(sev_counts, height=200)

        st.subheader("🕐 Today vs Yesterday")
        compare = pd.DataFrame({
            "Intrusions": {"Today": len(df_today), "Yesterday": len(df_yest)}
        })
        st.bar_chart(compare, height=180)

    st.divider()
    st.download_button("⬇  Download Full Log as CSV",
                       df.to_csv(index=False).encode(),
                       "intrusion_log.csv", "text/csv")

# ═══════════════════════════════════════════════════════
# PAGE: EVIDENCE
# ═══════════════════════════════════════════════════════
elif menu == "🗂 Evidence":

    st.markdown("## 🗂 EVIDENCE VAULT")

    imgs = sorted(
        [f for f in os.listdir("intruders") if f.lower().endswith((".jpg",".jpeg",".png"))],
        reverse=True
    )

    if not imgs:
        st.markdown('<div class="alert-box ok">📂 No evidence captured yet. Start Live Surveillance to collect images.</div>',
                    unsafe_allow_html=True)
    else:
        # ── Top toolbar ──
        toolbar_l, toolbar_r = st.columns([3, 1])
        with toolbar_l:
            st.markdown(f'<div class="alert-box warn">📸 <b>{len(imgs)}</b> image(s) on record</div>',
                        unsafe_allow_html=True)
        with toolbar_r:
            if st.button("🗑️ CLEAR ALL IMAGES", key="clear_all_evidence"):
                st.session_state["confirm_clear_all"] = True

        # ── Confirm dialog for bulk clear ──
        if st.session_state.get("confirm_clear_all"):
            st.markdown('<div class="alert-box">⚠️ This will permanently delete ALL captured evidence images. This cannot be undone.</div>',
                        unsafe_allow_html=True)
            ca, cb, _ = st.columns([1, 1, 4])
            with ca:
                if st.button("✅ YES, DELETE ALL", key="confirm_yes"):
                    deleted = 0
                    for f in os.listdir("intruders"):
                        if f.lower().endswith((".jpg",".jpeg",".png")):
                            os.remove(os.path.join("intruders", f))
                            deleted += 1
                    st.session_state["confirm_clear_all"] = False
                    st.success(f"🗑️ Deleted {deleted} image(s).")
                    st.rerun()
            with cb:
                if st.button("✖ CANCEL", key="confirm_no"):
                    st.session_state["confirm_clear_all"] = False
                    st.rerun()

        st.divider()

        # ── Filter bar ──
        def parse_type(name):
            parts = name.rsplit(".", 1)[0].split("_")
            return parts[2] if len(parts) >= 3 else "UNKNOWN"

        def parse_severity(name):
            parts = name.rsplit(".", 1)[0].split("_")
            return parts[3] if len(parts) >= 4 else "—"

        types_avail = sorted(set(parse_type(i) for i in imgs))
        fc1, fc2 = st.columns(2)
        with fc1:
            flt_type = st.selectbox("🔍 Filter by Type", ["ALL"] + types_avail, key="ev_flt_type")
        with fc2:
            flt_sev = st.selectbox("⚡ Filter by Severity", ["ALL", "HIGH", "MEDIUM", "LOW"], key="ev_flt_sev")

        filtered = imgs
        if flt_type != "ALL":
            filtered = [i for i in filtered if parse_type(i) == flt_type]
        if flt_sev != "ALL":
            filtered = [i for i in filtered if parse_severity(i) == flt_sev]

        st.markdown(
            f"<div class='section-title'>SHOWING {len(filtered)} OF {len(imgs)} RECORD(S)</div>",
            unsafe_allow_html=True
        )

        if not filtered:
            st.markdown('<div class="alert-box info">No images match the selected filters.</div>',
                        unsafe_allow_html=True)
        else:
            # ── Image grid — 3 columns, each with Download + Delete ──
            cols = st.columns(3)
            for idx, img_name in enumerate(filtered[:48]):
                parts   = img_name.rsplit(".", 1)[0].split("_")
                obj_type = parts[2] if len(parts) >= 3 else "?"
                severity = parts[3] if len(parts) >= 4 else ""
                ts_raw   = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else img_name
                try:
                    ts_fmt = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S").strftime("%d %b %Y  %H:%M:%S")
                except Exception:
                    ts_fmt = ts_raw

                sev_color = {"HIGH": "var(--red)", "MEDIUM": "var(--yellow)", "LOW": "var(--accent)"}.get(severity, "var(--dim)")
                caption_html = (
                    f"<div style='font-family:var(--font-mono);font-size:0.65rem;padding:4px 0 2px;"
                    f"color:var(--dim);letter-spacing:1px;'>{ts_fmt}</div>"
                    f"<div style='display:flex;gap:6px;align-items:center;margin-bottom:4px;'>"
                    f"<span class='badge badge-{obj_type}'>{obj_type}</span>"
                    f"<span style='font-family:var(--font-mono);font-size:0.62rem;color:{sev_color};'>{severity}</span>"
                    f"</div>"
                )
                path = os.path.join("intruders", img_name)
                with cols[idx % 3]:
                    st.image(path, use_container_width=True)
                    st.markdown(caption_html, unsafe_allow_html=True)
                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        with open(path, "rb") as fh:
                            st.download_button(
                                "⬇ Save", fh.read(), img_name, "image/jpeg",
                                key=f"dl_{img_name}", use_container_width=True
                            )
                    with btn_c2:
                        if st.button("🗑 Delete", key=f"del_{img_name}", use_container_width=True):
                            os.remove(path)
                            st.success(f"Deleted {img_name}")
                            st.rerun()
                    st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════
elif menu == "⚙️ Settings":

    st.markdown("## ⚙️ SYSTEM SETTINGS")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### System Info")
        st.markdown(f"""
        <div class='kpi-card' style='text-align:left;padding:18px 20px;'>
        <p style='font-family:Share Tech Mono,monospace;font-size:0.8rem;line-height:2;color:var(--accent);'>
        OS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {platform.system()} {platform.release()}<br>
        Python &nbsp;&nbsp;: {sys.version.split()[0]}<br>
        OpenCV &nbsp;&nbsp;: {cv2.__version__}<br>
        Model &nbsp;&nbsp;&nbsp;: {model_choice}<br>
        YOLO OK &nbsp;: {"LOADED" if model else "FAILED"}<br>
        Log File : {LOG_FILE}<br>
        Snapshots: {len(os.listdir("intruders"))} files
        </p></div>""", unsafe_allow_html=True)

        st.markdown("### Export Log")
        df_exp = read_log()
        if not df_exp.empty:
            st.download_button("⬇ Download CSV", df_exp.to_csv(index=False).encode(),
                               "intrusion_log.csv","text/csv")
        else:
            st.markdown('<div class="alert-box info">No log data to export.</div>',
                        unsafe_allow_html=True)

    with c2:
        st.markdown("### Reset Options")
        st.markdown('<div class="alert-box warn">⚠ These actions cannot be undone.</div>',
                    unsafe_allow_html=True)

        if st.button("🗑 CLEAR ALL SNAPSHOTS"):
            for f in os.listdir("intruders"):
                os.remove(os.path.join("intruders",f))
            st.success("All snapshots deleted.")

        if st.button("🗑 RESET INTRUSION LOG"):
            pd.DataFrame(columns=COLS).to_csv(LOG_FILE, index=False)
            st.session_state.total_today  = 0
            st.session_state.zone_breaches = 0
            st.session_state.live_log.clear()
            st.success("Log reset.")

        st.markdown("### Accuracy Tips")
        st.markdown("""
        <div class='alert-box info' style='line-height:2;'>
        Use <b>yolov8s</b> or <b>yolov8m</b> for best accuracy<br>
        Confidence threshold: <b>0.30 – 0.45</b> recommended<br>
        Enable <b>Enhance Low-Light</b> in dim environments<br>
        Inference size <b>640</b> for best small-object detection<br>
        Temporal Smoothing <b>2–3</b> removes single-frame false alerts<br>
        Ensure camera is at 720p+ with good lighting
        </div>""", unsafe_allow_html=True)