
"""Standalone local monitor: python main.py --source video.mp4"""
import argparse
import csv
from pathlib import Path
import cv2
from ultralytics import YOLO
from core import COLUMNS, process_frame
from analyze import prepare_data, generate_report
import pandas as pd


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='background_video_people_walking_720p.mp4')
    parser.add_argument('--weights', default='yolov8n.pt')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--safe-max', type=int, default=9)
    parser.add_argument('--moderate-max', type=int, default=19)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--repeat', action='store_true', help='Loop video explicitly')
    parser.add_argument('--no-display', action='store_true', help='Run without GUI')
    args = parser.parse_args()
    if args.moderate_max <= args.safe_max or args.safe_max < 0 or args.max_frames < 1:
        parser.error('Invalid thresholds or frame limit')
    source = 0 if args.source == '0' else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video source: {source}')
    try:
        model = YOLO(args.weights)
        records = []
        with open('crowd_data.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            while len(records)//3 < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    if args.repeat and source != 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok:
                        break
                annotated, rows, _, _ = process_frame(frame, model, args.confidence,
                    len(records)//3, args.safe_max, args.moderate_max)
                writer.writerows(rows)
                records.extend(rows)
                if not args.no_display:
                    cv2.imshow('SentinelFlow — ESC to exit', annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        if records:
            report = generate_report(prepare_data(pd.DataFrame(records)))
            Path('reports.txt').write_text(report, encoding='utf-8')
            print(f'Processed {len(records)//3} frames; saved crowd_data.csv and reports.txt')
        else:
            print('No readable frames found.')
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
