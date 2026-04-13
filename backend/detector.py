import cv2
import time
import asyncio
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, HARMFUL_CLASSES
from utils import save_screenshot, log_alert, get_current_timestamp


class VideoDetector:
    def __init__(self):
        self.running = False
        self.cap = None
        self.model = None
        self.alert_callbacks = []

        try:
            self.model = YOLO(MODEL_PATH)
            print(f"Loaded YOLO model from {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load model from {MODEL_PATH}: {e}")

    def add_alert_callback(self, callback):
        self.alert_callbacks.append(callback)

    async def _emit_alerts(self, alerts):
        for callback in self.alert_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(alerts)
            else:
                callback(alerts)

    async def start(self, source):
        if self.running:
            print("Detector is already running.")
            return

        # Handle webcam source index
        if str(source).isdigit():
            source = int(source)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

        self.running = True
        print(f"Started detection on source: {source}")

        fps_start_time = time.time()
        fps_frame_count = 0

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame, or end of stream.")
                break

            fps_frame_count += 1
            if time.time() - fps_start_time > 1:
                # fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Run inference in a separate thread to unblock the async loop
            if self.model:
                results = await asyncio.to_thread(self.model.predict, frame, verbose=False)
                frame_alerts = []
                timestamp = get_current_timestamp()

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        # Map index to class name
                        class_name = self.model.names.get(class_id, f"class_{class_id}")

                        if class_name in HARMFUL_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                            bbox = box.xyxy[0].tolist()

                            # Draw bbox internally or just rely on saving screenshot
                            # Optional: cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

                            save_screenshot(frame, class_name, timestamp)
                            log_alert(timestamp, class_name, conf, bbox)

                            alert_data = {
                                "class": class_name,
                                "confidence": conf,
                                "timestamp": timestamp,
                            }
                            frame_alerts.append(alert_data)
                            print(f"[ALERT] {class_name} detected with conf {conf:.2f}")

                if frame_alerts:
                    await self._emit_alerts(frame_alerts)

            # Allow other tasks to run
            await asyncio.sleep(0.01)

        self.stop()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Stopped detection loop.")
