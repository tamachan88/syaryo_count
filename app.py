from flask import Flask, request, render_template, url_for
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_VIDEO = 'static/output.mp4'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

class SimpleTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}  
        self.max_distance = max_distance
        self.previous_positions = {}  

    def update(self, detections):
        if len(detections) == 0:
            self.objects = {}
            self.previous_positions = {}
            return []

        centers = np.array([[(x1+x2)/2, (y1+y2)/2] for (x1, y1, x2, y2) in detections])
        updated_objects = {}
        assigned_ids = []

        if len(self.objects) == 0:
            for i, bbox in enumerate(detections):
                updated_objects[self.next_id] = centers[i]
                self.previous_positions[self.next_id] = centers[i][0]
                assigned_ids.append((self.next_id, bbox))
                self.next_id += 1
            self.objects = updated_objects
            return assigned_ids

        existing_ids = list(self.objects.keys())
        existing_centers = np.array([self.objects[obj_id] for obj_id in existing_ids])
        dist_matrix = np.linalg.norm(existing_centers[:, None, :] - centers[None, :, :], axis=2)

        used_existing = set()
        used_new = set()
        while True:
            if dist_matrix.size == 0:
                break
            min_idx = np.unravel_index(np.argmin(dist_matrix, axis=None), dist_matrix.shape)
            dist = dist_matrix[min_idx]
            if dist > self.max_distance:
                break
            e_idx, d_idx = min_idx
            if e_idx in used_existing or d_idx in used_new:
                dist_matrix[e_idx, d_idx] = np.inf
                continue
            obj_id = existing_ids[e_idx]
            updated_objects[obj_id] = centers[d_idx]
            assigned_ids.append((obj_id, detections[d_idx]))
            used_existing.add(e_idx)
            used_new.add(d_idx)
            dist_matrix[e_idx, :] = np.inf
            dist_matrix[:, d_idx] = np.inf

        for i, bbox in enumerate(detections):
            if i not in used_new:
                updated_objects[self.next_id] = centers[i]
                self.previous_positions[self.next_id] = centers[i][0]
                assigned_ids.append((self.next_id, bbox))
                self.next_id += 1

        self.objects = updated_objects
        return assigned_ids

@app.route('/')
def index():
    return render_template('index.html', video_url=None, count=None)

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
    input_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(input_path)
    count = process_video(input_path, OUTPUT_VIDEO)
    return render_template('index.html', video_url=url_for('static', filename='output.mp4'), count=count)

def process_video(input_path, output_path):
    model = YOLO("yolo11s_openvino_model/")
    cap = cv2.VideoCapture(input_path)
    tracker = SimpleTracker(max_distance=50)

    counted_ids = set()
    count_left_to_right = 0
    count_right_to_left = 0
    vehicle_classes = {2, 3, 5, 7}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    line_x = w // 2 + 50
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, device="intel:gpu")[0]
        dets = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in vehicle_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if box.conf[0].item() < 0.3:
                continue
            dets.append([x1, y1, x2, y2])

        tracked_objects = tracker.update(dets)

        for obj_id, bbox in tracked_objects:
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            if obj_id in counted_ids:
                continue
            prev_x = tracker.previous_positions.get(obj_id, None)
            if prev_x is not None:
                if prev_x < line_x and cx >= line_x:
                    count_left_to_right += 1
                    counted_ids.add(obj_id)
                elif prev_x > line_x and cx <= line_x:
                    count_right_to_left += 1
                    counted_ids.add(obj_id)
            tracker.previous_positions[obj_id] = cx
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)
        cv2.putText(frame, f"L->R: {count_left_to_right}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"R->L: {count_right_to_left}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    return count_left_to_right + count_right_to_left

if __name__ == '__main__':
    app.run(debug=True)
