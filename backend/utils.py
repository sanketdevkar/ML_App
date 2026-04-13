import cv2
import pandas as pd
import os
from datetime import datetime
from config import SCREENSHOTS_DIR, ALERTS_CSV


def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_screenshot(frame, class_name, timestamp):
    """
    Saves the given frame as a screenshot.
    """
    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
    filename = f"{class_name}_{safe_ts}.jpg"
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filename


def log_alert(timestamp, class_name, confidence, bbox):
    """
    Logs an alert to the CSV file.
    """
    new_data = pd.DataFrame([{
        "timestamp": timestamp,
        "class": class_name,
        "confidence": round(confidence, 2),
        "bbox": str(bbox)
    }])
    new_data.to_csv(ALERTS_CSV, mode='a', header=False, index=False)
