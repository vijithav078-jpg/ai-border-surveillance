import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import csv
import os
import winsound  # for sound (Windows)

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# Zone
zone_top = 250
zone_bottom = 400

intrusion_count = 0
last_alert_time = 0
running = True

# Create folders
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Create log file
if not os.path.exists("intrusion_log.csv"):
    with open("intrusion_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Object", "Status"])

print("Press 's' to START/STOP, 'q' to EXIT")

while True:

    if running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Draw zone
        cv2.rectangle(frame, (0, zone_top), (640, zone_bottom), (0, 0, 255), 2)
        cv2.putText(frame, "RESTRICTED BORDER ZONE", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if label != "person":
                    continue

                center_y = (y1 + y2) // 2

                # Classification
                if zone_top < center_y < zone_bottom:
                    status = "UNAUTHORIZED"
                    color = (0, 0, 255)
                else:
                    status = "AUTHORIZED"
                    color = (0, 255, 0)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} - {status}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2)

                # Intrusion alert
                if status == "UNAUTHORIZED":
                    current_time = time.time()

                    if current_time - last_alert_time > 3:
                        intrusion_count += 1
                        last_alert_time = current_time

                        # 🔊 SOUND ALERT
                        winsound.Beep(1000, 500)

                        # 📸 SAVE IMAGE
                        filename = f"outputs/intruder_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)

                        # 📊 LOG DATA
                        with open("intrusion_log.csv", "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([datetime.now(), label, status])

                    cv2.putText(frame, "⚠ INTRUSION ALERT",
                                (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                3)

        # Show count
        cv2.putText(frame, f"Intrusions: {intrusion_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2)

        cv2.imshow("AI Border Surveillance System", frame)

    # 🎮 KEY CONTROLS
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('s'):
        running = not running
        print("Running:", running)

cap.release()
cv2.destroyAllWindows()