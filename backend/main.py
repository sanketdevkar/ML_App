import os
import pandas as pd

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from detector import VideoDetector
from config import ALERTS_CSV, SCREENSHOTS_DIR

app = FastAPI(title="Harmful Object Detection API")
detector = VideoDetector()

# List of active WebSocket connections
connected_clients = []


async def broadcast_alert(alerts):
    """
    Callback function that acts as a bridge between the detector and websocket clients.
    """
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(alerts)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)

detector.add_alert_callback(broadcast_alert)


class StartRequest(BaseModel):
    source: str = Field(
        ...,
        min_length=1,
        description="Video source like '0' (webcam), a local file path, or an RTSP stream URL"
    )


@app.post("/start")
async def start_detection(request: StartRequest, background_tasks: BackgroundTasks):
    if detector.running:
        return {"status": "error", "message": "Detection is already running."}

    # We use a wrapper string for source since FastAPI payload is JSON
    background_tasks.add_task(detector.start, request.source)
    return {"status": "success", "message": f"Detection started on source: {request.source}"}


@app.post("/stop")
async def stop_detection():
    if not detector.running:
        return {"status": "error", "message": "Detection is not running."}

    detector.stop()
    return {"status": "success", "message": "Detection stopped."}


@app.get("/alerts")
async def get_alerts():
    if not os.path.exists(ALERTS_CSV):
        return []

    try:
        df = pd.read_csv(ALERTS_CSV)
        # Return records formatted as dictionary
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.get("/screenshots")
async def get_screenshots():
    if not os.path.exists(SCREENSHOTS_DIR):
        return {"screenshots": []}

    files = os.listdir(SCREENSHOTS_DIR)
    # Return a list of available screenshot names
    return {"screenshots": files}


@app.get("/screenshots/{filename}")
async def get_screenshot_file(filename: str):
    """
    Provides a way to retrieve the actual image file via browser or frontend.
    """
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return {"error": "File not found"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        # Keep connection open loop
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
