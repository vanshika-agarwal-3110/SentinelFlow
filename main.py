import cv2
from ultralytics import YOLO
import csv
from datetime import datetime
import os
 
# Use a local video file or set to 0 for a connected camera feed
#INPUT_PATH = "background_video_people_walking_720p.mp4"
# Reads whatever option was clicked on the frontend app.py page
env_source = os.getenv("SENTINEL_SOURCE", "background_video_people_walking_720p.mp4")
# Convert string '0' to integer 0 if the user chose the live camera feed option
INPUT_PATH = int(env_source) if env_source == "0" else env_source
CONFIDENCE_THRESHOLD = 0.25  # Balanced for YOLOv8s
DISPLAY_RES = (960, 720)

# Load the S-model for a good balance between speed and accuracy
model = YOLO("yolov8s.pt")

# Initialize capture and verify source
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    print(f"Error: Unable to access video source at {INPUT_PATH}")
    exit()

# Define monitoring zones (Spatial Analysis)
# Structure: [x1, y1, x2, y2]
zones = [
    {"name": "Left Area", "coords": (0, 0, 320, 720), "count": 0},
    {"name": "Central Area", "coords": (320, 0, 640, 720), "count": 0},
    {"name": "Right Area", "coords": (640, 0, 960, 720), "count": 0}
]

# Calculate pixel area for density math
for z in zones:
    x1, y1, x2, y2 = z["coords"]
    z["total_pixels"] = (x2 - x1) * (y2 - y1)

# Initialize CSV log file with headers if it's a new run
log_path = "crowd_data.csv"
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Zone", "Count", "Density", "Status"])

while True:
    success, frame = cap.read()
    
    # Simple loop for demonstration videos
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, DISPLAY_RES)
    
    # Run YOLO inference (person class only - index 0)
    results = model(frame, verbose=False)
    
    # Reset counts for the current frame
    for z in zones:
        z["count"] = 0

    # Process detections
    for r in results:
        for box in r.boxes:
            label = int(box.cls[0])
            score = float(box.conf[0])

            if label == 0 and score > CONFIDENCE_THRESHOLD:
                # Get bounding box centers
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Assign person to a zone based on centroid location
                matched_zone = False
                for z in zones:
                    zx1, zy1, zx2, zy2 = z["coords"]
                    if zx1 < cx < zx2 and zy1 < cy < zy2:
                        z["count"] += 1
                        matched_zone = True
                
                # Visual feedback: Green if in a zone, red if outside monitored areas
                rect_color = (0, 255, 0) if matched_zone else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)

    # Risk Assessment and Data Logging
    timestamp = datetime.now().strftime("%H:%M:%S")
    frame_severity = "LOW"
    
    log_entries = []

    for z in zones:
        density = z["count"] / z["total_pixels"]
        
        # Density-based status thresholds
        if density < 0.00002:
            z_status, z_color = "SAFE", (0, 255, 0)
        elif density < 0.00005:
            z_status, z_color = "MODERATE", (0, 255, 255)
            if frame_severity != "HIGH": frame_severity = "MODERATE"
        else:
            z_status, z_color = "DANGER", (0, 0, 255)
            frame_severity = "HIGH"

        z["status"] = z_status
        zx1, zy1, zx2, zy2 = z["coords"]
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), z_color, 1)

        log_entries.append([timestamp, z["name"], z["count"], round(density, 6), z_status])

    # Batch write to CSV to reduce disk I/O overhead
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerows(log_entries)

    # UI Overlay - Semi-transparent background
    ui_overlay = frame.copy()
    cv2.rectangle(ui_overlay, (10, 10), (330, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(ui_overlay, 0.6, frame, 0.4, 0)

    y_pos = 40
    for z in zones:
        status_text = f"{z['name']}: {z['count']} ({z['status']})"
        cv2.putText(frame, status_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

    total_count = sum(z["count"] for z in zones)
    cv2.putText(frame, f"System Total: {total_count}", (20, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    risk_color = (0, 0, 255) if frame_severity == "HIGH" else (255, 255, 255)
    cv2.putText(frame, f"Risk Level: {frame_severity}", (20, y_pos + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)

    if frame_severity == "HIGH":
        cv2.putText(frame, "!!! CROWD LIMIT EXCEEDED !!!", (20, DISPLAY_RES[1] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if os.getenv("SENTINEL_SOURCE") is not None:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        cv2.imshow("SentinelFlow - Live Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()