import streamlit as st
import cv2
import time
from ultralytics import YOLO
import pandas as pd
import os
from datetime import datetime
from PIL import Image

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Army Surveillance System", layout="wide")

# -------- ARMY THEME --------
st.markdown("""
<style>
.stApp {
    background-color: #0b1a0b;
    color: #00ff9f;
}
h1, h2, h3 {
    color: #00ff9f;
    text-align: center;
}
.stButton>button {
    background-color: #1f3d1f;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------- TITLE --------
st.title("🪖 AI BORDER SURVEILLANCE SYSTEM")

# -------- NAVIGATION --------
page = st.sidebar.radio("Select Page", ["Live Detection", "Dashboard"])

# -------- LOAD MODEL --------
model = YOLO("yolov8n.pt")

if "running" not in st.session_state:
    st.session_state.running = False

# ================= LIVE DETECTION =================
if page == "Live Detection":

    st.subheader("🎥 Live Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Start"):
            st.session_state.running = True

    with col2:
        if st.button("⏹ Stop"):
            st.session_state.running = False

    FRAME_WINDOW = st.image([])

    line_y = 250
    intrusion_count = 0
    last_alert_time = 0

    if st.session_state.running:

        cap = cv2.VideoCapture(0)

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])

                    if cls == 0:  # person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                        cv2.circle(frame, (cx,cy),5,(0,0,255),-1)

                        if cy > line_y and time.time() - last_alert_time > 3:
                            intrusion_count += 1
                            last_alert_time = time.time()

                            if not os.path.exists("outputs"):
                                os.makedirs("outputs")

                            filename = f"outputs/intruder_{int(time.time())}.jpg"
                            cv2.imwrite(filename, frame)

                            with open("intrusion_log.csv","a",newline="") as f:
                                import csv
                                writer = csv.writer(f)
                                writer.writerow([datetime.now(), intrusion_count])

                            st.error("🚨 INTRUSION DETECTED 🚨")

            cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,255,0),2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()

# ================= DASHBOARD =================
elif page == "Dashboard":

    st.subheader("📊 Dashboard")

    if os.path.exists("intrusion_log.csv"):
        df = pd.read_csv("intrusion_log.csv", names=["Time","Count"])

        st.metric("Total Intrusions", int(df["Count"].iloc[-1]))
        st.line_chart(df["Count"])

    else:
        st.warning("No data yet")

    st.subheader("📸 Intrusion Images")

    if os.path.exists("outputs"):
        images = os.listdir("outputs")

        if images:
            images = sorted(images, reverse=True)

            cols = st.columns(3)

            for i, img_name in enumerate(images[:6]):
                img = Image.open(os.path.join("outputs", img_name))

                with cols[i % 3]:
                    st.image(img, caption=img_name)
        else:
            st.info("No images yet")