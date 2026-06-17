from flask import Flask, render_template, Response
import cv2
import time
import threading
from ultralytics import YOLO
from real import (
    MODEL_NAME, DETECT_EVERY_N_FRAMES, INFER_SIZE, CONFIDENCE_THRESHOLD, 
    THREAT_CLASSES, ALERT_COOLDOWN, send_whatsapp_with_photo, draw_box, log
)

app = Flask(__name__)
model = YOLO(MODEL_NAME)
last_alert_time = {}

def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_results = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            now = time.time()

            # DETECTION
            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                small_frame = cv2.resize(frame, INFER_SIZE)
                last_results = model(small_frame, verbose=False)

            if last_results:
                h_ratio = frame.shape[0] / INFER_SIZE[1]
                w_ratio = frame.shape[1] / INFER_SIZE[0]

                for r in last_results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf < CONFIDENCE_THRESHOLD:
                            continue

                        cls = int(box.cls[0])
                        name = model.names[cls]

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, x2 = int(x1 * w_ratio), int(x2 * w_ratio)
                        y1, y2 = int(y1 * h_ratio), int(y2 * h_ratio)

                        if name in THREAT_CLASSES:
                            if now - last_alert_time.get(name, 0) > ALERT_COOLDOWN:
                                threading.Thread(
                                    target=send_whatsapp_with_photo,
                                    args=(name, frame.copy()),
                                    daemon=True
                                ).start()
                                last_alert_time[name] = now

                            draw_box(frame, x1, y1, x2, y2,
                                     f"THREAT: {name} {conf:.2f}", (0, 0, 255))
                        else:
                            label = "SAFE PERSON" if name == "person" else f"SAFE: {name}"
                            draw_box(frame, x1, y1, x2, y2,
                                     f"{label} {conf:.2f}", (0, 255, 0))

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    log("Starting STARK Web Interface")
    app.run(host='0.0.0.0', port=5000, debug=True)
