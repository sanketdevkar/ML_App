import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Using relative path to model as requested
# Assuming the file will be in the backend folder or provided relative to it
MODEL_PATH = os.path.join(BASE_DIR, "HOD_6class_yolov8.pt")

CONFIDENCE_THRESHOLD = 0.5
HARMFUL_CLASSES = ["gun", "knife", "rifle", "syringe", "alcohol", "cigarette"]

LOGS_DIR = os.path.join(BASE_DIR, "logs")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
ALERTS_CSV = os.path.join(LOGS_DIR, "alerts.csv")

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Initialize alerts CSV if it doesn't exist
if not os.path.exists(ALERTS_CSV):
    df = pd.DataFrame(columns=["timestamp", "class", "confidence", "bbox"])
    df.to_csv(ALERTS_CSV, index=False)
