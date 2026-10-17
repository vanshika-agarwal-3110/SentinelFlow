
"""Shared detection, zone assignment, and prototype risk rules."""
from datetime import datetime
import cv2

WIDTH, HEIGHT = 960, 720
ZONES = ('Left Area', 'Central Area', 'Right Area')
COLUMNS = ['Timestamp', 'Frame', 'Zone', 'Count', 'Density', 'Status']
COLORS = {'SAFE': (0, 200, 0), 'MODERATE': (0, 190, 255), 'DANGER': (0, 0, 255)}

def status(count, safe_max=9, moderate_max=19):
    return 'SAFE' if count <= safe_max else 'MODERATE' if count <= moderate_max else 'DANGER'

def overall(statuses):
    return 'DANGER' if 'DANGER' in statuses else 'MODERATE' if 'MODERATE' in statuses else 'SAFE'

def process_frame(frame, model, confidence, frame_index, safe_max=9, moderate_max=19, show_conf=True):
    if frame is None or frame.size == 0:
        raise ValueError('Invalid video frame')
    # Letterbox rather than stretching the video, preserving aspect ratio.
    h, w = frame.shape[:2]
    scale = min(WIDTH / w, HEIGHT / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    frame = cv2.resize(frame, (nw, nh))
    canvas = cv2.copyMakeBorder(frame, (HEIGHT-nh)//2, HEIGHT-nh-(HEIGHT-nh)//2,
                                (WIDTH-nw)//2, WIDTH-nw-(WIDTH-nw)//2,
                                cv2.BORDER_CONSTANT, value=(18, 23, 30))
    result = model.predict(canvas, classes=[0], conf=confidence, verbose=False)[0]
    counts = [0, 0, 0]
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = max(0, min(WIDTH-1, (x1+x2)//2))
        i = min(2, cx * 3 // WIDTH)
        counts[i] += 1
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (60,220,130), 2)
        if show_conf:
            cv2.putText(canvas, f'Person {float(box.conf[0]):.2f}', (x1,max(15,y1-7)),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
    statuses = [status(c, safe_max, moderate_max) for c in counts]
    timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
    rows = []
    for i, (name, count, state) in enumerate(zip(ZONES, counts, statuses)):
        x1, x2 = i * WIDTH//3, (i+1)*WIDTH//3
        cv2.rectangle(canvas, (x1,0), (x2-1,HEIGHT-1), COLORS[state], 2)
        cv2.rectangle(canvas, (x1+3,5), (x2-4,43), (18,23,30), -1)
        cv2.putText(canvas, f'{name}: {count} {state}', (x1+9,30),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, COLORS[state], 2)
        rows.append(dict(Timestamp=timestamp, Frame=frame_index, Zone=name, Count=count,
                         Density=round(count / ((WIDTH//3)*HEIGHT), 8), Status=state))
    cv2.rectangle(canvas, (0,HEIGHT-45), (WIDTH,HEIGHT), (18,23,30), -1)
    cv2.putText(canvas, f'Total detected: {sum(counts)} | Risk: {overall(statuses)}',
                (15,HEIGHT-16), cv2.FONT_HERSHEY_SIMPLEX, .7,
                COLORS[overall(statuses)], 2)
    return canvas, rows, counts, statuses
