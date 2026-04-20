import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import time
import threading
import winsound

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Border Surveillance", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp {
    background-color: #031403;
    background-image:
      linear-gradient(rgba(0,255,120,0.08) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,255,120,0.08) 1px, transparent 1px);
    background-size: 46px 46px;
    color: #00ff9f;
}
h1,h2,h3 {
    color:#00ff9f;
    text-shadow:0 0 10px #00ff9f;
}
[data-testid="stSidebar"] {
    background:#020b02;
    border-right:1px solid #00ff9f55;
}
.stButton > button {
    background:#041104;
    color:#00ff9f;
    border:1px solid #00ff9f;
    border-radius:10px;
    font-weight:bold;
}
.stButton > button:hover {
    box-shadow:0 0 15px #00ff9f;
}
.block {
    border:1px solid #00ff9f55;
    border-radius:14px;
    padding:14px;
    background:rgba(0,255,159,0.03);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🛡️ AI BORDER SURVEILLANCE SYSTEM</h1>", unsafe_allow_html=True)

# ---------------- FILES ----------------
os.makedirs("intruders", exist_ok=True)

if not os.path.exists("intrusion_log.csv"):
    with open("intrusion_log.csv", "w") as f:
        f.write("Time,Type,Status\n")

# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- SETTINGS ----------------
zone_top = 220
zone_bottom = 420

# categories
human_classes = ["person"]
animal_classes = ["dog", "cat", "cow", "horse", "sheep", "bird"]
vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
electronic_classes = [
    "laptop", "cell phone", "tv", "remote",
    "keyboard", "mouse"
]

# ---------------- ALARM ----------------
alarm_playing = False

def play_alarm():
    global alarm_playing
    alarm_playing = True
    for _ in range(5):     # ~4 sec
        winsound.Beep(1200, 300)
        time.sleep(0.4)
    alarm_playing = False

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "CONTROL PANEL",
    ["🎥 Live Surveillance", "📊 Dashboard"]
)

# =====================================================
# LIVE SURVEILLANCE
# =====================================================
if menu == "🎥 Live Surveillance":

    st.markdown("### 🎥 LIVE CAMERA FEED")

    c1, c2 = st.columns([1,1])

    with c1:
        start = st.button("▶ START")

    with c2:
        stop = st.button("⛔ STOP")

    if "run" not in st.session_state:
        st.session_state.run = False

    if start:
        st.session_state.run = True

    if stop:
        st.session_state.run = False

    frame_placeholder = st.empty()

    if st.session_state.run:

        cap = cv2.VideoCapture(0)
        last_detection = 0

        while st.session_state.run:

            ret, frame = cap.read()

            if not ret:
                st.error("Camera not detected")
                break

            # draw restricted zone
            cv2.rectangle(
                frame,
                (0, zone_top),
                (640, zone_bottom),
                (0,0,255),
                2
            )

            cv2.putText(
                frame,
                "RESTRICTED ZONE",
                (20, zone_top - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

            # detect
            results = model(frame, verbose=False)

            for r in results:
                for box in r.boxes:

                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # box
                    cv2.rectangle(
                        frame,
                        (x1,y1),
                        (x2,y2),
                        (0,255,0),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2
                    )

                    center_y = (y1+y2)//2

                    # if enters zone
                    if zone_top < center_y < zone_bottom:

                        # classify intrusion type
                        if label in human_classes:
                            intrusion_type = "PERSON"

                        elif label in animal_classes:
                            intrusion_type = "ANIMAL"

                        elif label in vehicle_classes:
                            intrusion_type = "VEHICLE"

                        elif label in electronic_classes:
                            intrusion_type = "ELECTRONIC"

                        else:
                            intrusion_type = label.upper()

                        current = time.time()

                        # avoid repeated spam
                        if current - last_detection > 5:
                            last_detection = current

                            # play alarm once
                            if not alarm_playing:
                                threading.Thread(
                                    target=play_alarm,
                                    daemon=True
                                ).start()

                            # warning text
                            cv2.putText(
                                frame,
                                f"{intrusion_type} INTRUSION",
                                (20,60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,0,255),
                                3
                            )

                            # save image
                            fname = datetime.now().strftime(
                                "%Y%m%d_%H%M%S"
                            )

                            img_path = f"intruders/{fname}_{intrusion_type}.jpg"
                            cv2.imwrite(img_path, frame)

                            # log csv
                            with open("intrusion_log.csv", "a") as f:
                                f.write(
                                    f"{datetime.now()},{intrusion_type},INTRUSION\n"
                                )

            # show frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()

# =====================================================
# DASHBOARD
# =====================================================
if menu == "📊 Dashboard":

    st.markdown("## 📊 INTELLIGENCE DASHBOARD")

    try:
        df = pd.read_csv("intrusion_log.csv", on_bad_lines="skip")

        if not df.empty:

            # clean time
            df["Time"] = pd.to_datetime(
                df["Time"],
                errors="coerce"
            )

            df = df.dropna(subset=["Time"])

            # metric
            st.metric("🚨 Total Intrusions", len(df))

            # table
            st.dataframe(df, use_container_width=True)

            # hourly graph
            df["Hour"] = df["Time"].dt.hour
            hourly = df.groupby("Hour").size()
            hourly = hourly.reindex(range(24), fill_value=0)

            st.subheader("📈 Intrusion Frequency (Hourly)")
            st.area_chart(hourly)

            # type graph
            st.subheader("📦 Intrusion Types")
            type_counts = df["Type"].value_counts()
            st.bar_chart(type_counts)

        else:
            st.info("No data yet")

    except Exception as e:
        st.error("CSV Error")
        st.text(str(e))

    # ---------------- IMAGES ----------------
    st.subheader("📸 Captured Intruders")

    imgs = sorted(os.listdir("intruders"), reverse=True)

    if imgs:
        for img in imgs[:8]:
            st.image(
                os.path.join("intruders", img),
                caption=img,
                use_container_width=True
            )
    else:
        st.info("No images yet")

    # ---------------- CLEAR ----------------
    if st.button("🗑 CLEAR ALL DATA"):

        # clear images
        for img in os.listdir("intruders"):
            os.remove(os.path.join("intruders", img))

        # reset csv
        with open("intrusion_log.csv", "w") as f:
            f.write("Time,Type,Status\n")

        st.success("All data cleared")