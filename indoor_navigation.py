import cv2
import torch
import numpy as np
import time
import threading
import queue
import pyttsx3
import gc
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLOWorld
from transformers import pipeline

# ---------------- CONFIGURATION ----------------

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE = 0.15
VOICE_COOLDOWN = 4.0
PROCESS_EVERY_N_FRAMES = 3  # Only run models every 3rd frame to save GPU/Heat

YOLO_MODEL = Path(__file__).resolve().parent / "yolov8s-worldv2.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

CLASSES = [
    "person", "chair", "table", "room door",
    "stairs", "wall", "cabinet", "bed", "window"
]

OBSTACLES = {
    "person", "chair", "table", "cabinet", "bed", "wall"
}

PHRASES = {
    "STOP": "Stop. Obstacle ahead.",
    "MOVE LEFT": "Move right to avoid the obstacle on your left.",
    "MOVE RIGHT": "Move left to avoid the obstacle on your right.",
    "KEEP LEFT": "Keep left.",
    "KEEP RIGHT": "Keep right.",
    "CAUTION": "Caution. Move carefully.",
    "CONTINUE": "Continue carefully."
}

PRIORITY = {
    "CONTINUE": 1,
    "CAUTION": 2,
    "KEEP LEFT": 3,
    "KEEP RIGHT": 3,
    "MOVE LEFT": 4,
    "MOVE RIGHT": 4,
    "STOP": 5
}

# ---------------- VOICE GUIDANCE ----------------

speech_queue = queue.Queue(maxsize=1)
last_action = None
last_speech_time = 0.0
voice_shutdown = False
voice_lock = threading.Lock()

def voice_worker():
    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        while True:
            phrase = speech_queue.get()
            try:
                if phrase == "__STOP_VOICE_WORKER__":
                    return
                engine.say(phrase)
                engine.runAndWait()
            except Exception as exc:
                print("Voice error:", exc)
            finally:
                speech_queue.task_done()
    except Exception as exc:
        print("Voice initialization error:", exc)
    finally:
        if engine is not None:
            engine.stop()

def speak_navigation(action):
    global last_action, last_speech_time
    now = time.monotonic()
    
    with voice_lock:
        if voice_shutdown:
            return
        # 1. If it's the exact same action, respect the 4-second cooldown
        if action == last_action and (now - last_speech_time) < VOICE_COOLDOWN:
            return
            
        last_prio = PRIORITY.get(last_action, 0)
        new_prio = PRIORITY.get(action, 0)
        
        # 2. BUG FIX: If danger decreases (e.g., STOP -> CONTINUE), enforce cooldown.
        # This prevents the system from spamming "Continue" if a bounding box drops for 1 frame.
        # However, if danger INCREASES (CONTINUE -> STOP), it speaks immediately.
        if new_prio < last_prio and (now - last_speech_time) < VOICE_COOLDOWN:
            return

        phrase = PHRASES.get(action)
        if phrase is None:
            return
        last_action = action
        last_speech_time = now
        try:
            while True:
                speech_queue.get_nowait()
                speech_queue.task_done()
        except queue.Empty:
            pass
        speech_queue.put_nowait(phrase)

def stop_voice_worker():
    global voice_shutdown
    with voice_lock:
        voice_shutdown = True
        try:
            while True:
                speech_queue.get_nowait()
                speech_queue.task_done()
        except queue.Empty:
            pass
        speech_queue.put_nowait("__STOP_VOICE_WORKER__")

# ---------------- NAVIGATION HELPERS ----------------

def get_position(center_x, width):
    if center_x < width / 3:
        return "LEFT"
    elif center_x < 2 * width / 3:
        return "CENTER"
    return "RIGHT"

def distance_level(distance):
    if distance < 0.8:
        return "VERY NEAR"
    elif distance < 1.5:
        return "NEAR"
    elif distance < 3.0:
        return "MEDIUM"
    return "FAR"

def get_object_depth(depth_map, x1, y1, x2, y2):
    height, width = depth_map.shape[:2]

    x1 = max(0, min(int(x1), width - 1))
    x2 = max(0, min(int(x2), width))
    y1 = max(0, min(int(y1), height - 1))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    # Tighter crop: Use only the central 40% of the bounding box to avoid background bleed
    cx1 = int(x1 + 0.30 * box_w)
    cx2 = int(x2 - 0.30 * box_w)
    cy1 = int(y1 + 0.30 * box_h)
    cy2 = int(y2 - 0.30 * box_h)

    roi = depth_map[cy1:cy2, cx1:cx2]

    if roi.size == 0:
        return None

    valid = roi[np.isfinite(roi) & (roi > 0)]

    if valid.size == 0:
        return None

    # Get the 10th percentile (the closest parts of the object) instead of the median
    return float(np.percentile(valid, 10))

def navigation_decision(name, distance, position):
    # Immediate STOP if anything is dangerously close directly in front
    if distance < 1.0 and position == "CENTER":
        return "STOP"

    if name in {"room door", "window"}:
        if distance < 1.0: return "STOP"
        if distance < 2.0: return "CAUTION"
        return "CONTINUE"

    if name == "stairs":
        if distance < 1.5: return "STOP"
        if distance < 2.5: return "CAUTION"
        return "CONTINUE"

    if name not in OBSTACLES:
        return "CONTINUE"

    if distance < 1.3:
        if position == "CENTER": return "STOP"
        elif position == "LEFT": return "MOVE RIGHT"
        return "MOVE LEFT"

    if distance < 2.2:
        if position == "CENTER": return "CAUTION"
        elif position == "LEFT": return "KEEP RIGHT"
        return "KEEP LEFT"

    if distance < 3.5 and position == "CENTER":
        return "CAUTION"

    return "CONTINUE"

# ---------------- DEPTH CONVERSION ----------------

def get_depth_array(result):
    predicted = result.get("predicted_depth")
    if predicted is None:
        raise RuntimeError("Depth model did not return predicted_depth.")
    if isinstance(predicted, torch.Tensor):
        depth = predicted.detach().float().squeeze().cpu().numpy()
    else:
        depth = np.asarray(predicted, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise RuntimeError(f"Invalid depth map shape: {depth.shape}")
    return depth

# ---------------- MAIN APPLICATION ----------------

def main():
    device = 0 if torch.cuda.is_available() else -1
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if device == 0:
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU. Processing may be slow.")

    print("Loading YOLO-World...")
    detector = YOLOWorld(str(YOLO_MODEL))
    detector.set_classes(CLASSES)

    print("Loading Depth Anything V2...")
    depth_estimator = pipeline(
        "depth-estimation",
        model=DEPTH_MODEL,
        device=device
    )

    print("Models loaded.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open webcam. Check camera permissions or try CAMERA_INDEX = 1.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    threading.Thread(target=voice_worker, daemon=True).start()

    previous_time = time.monotonic()
    frame_count = 0
    
    detections_to_draw = []
    current_action = "CONTINUE"
    current_warning = ""
    visual = None

    try:
        with torch.inference_mode(): 
            while True:
                success, frame = cap.read()
                if not success:
                    print("Cannot read webcam frame.")
                    break

                frame_count += 1
                frame_h, frame_w = frame.shape[:2]

                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    results = detector.predict(source=frame, conf=CONFIDENCE, device=device, verbose=False)
                    result = results[0]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_small = cv2.resize(rgb, (320, 240))
                    depth_result = depth_estimator(Image.fromarray(rgb_small))
                    depth_map = get_depth_array(depth_result)
                    depth_map = cv2.resize(depth_map, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

                    current_action = "CONTINUE"
                    current_warning = ""
                    detections_to_draw = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
                            confidence = float(box.conf[0].detach().cpu())
                            class_id = int(box.cls[0].detach().cpu())
                            
                            names = result.names
                            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]

                            center_x = (x1 + x2) // 2
                            position = get_position(center_x, frame_w)
                            distance = get_object_depth(depth_map, x1, y1, x2, y2)

                            if distance is None:
                                continue

                            level = distance_level(distance)
                            action = navigation_decision(class_name, distance, position)

                            if PRIORITY.get(action, 0) > PRIORITY.get(current_action, 0):
                                current_action = action
                                current_warning = f"{action}: {class_name} {distance:.2f}m {position}"

                            detections_to_draw.append((x1, y1, x2, y2, class_name, confidence, distance, level, action, position))

                    # BUG FIX: Pure-Depth Safety Net for unclassified walls/close objects
                    # This bypasses YOLO entirely. If anything physical is <1.0m away in the center, it forces a STOP.
                    center_crop = depth_map[int(frame_h*0.3):int(frame_h*0.7), int(frame_w*0.3):int(frame_w*0.7)]
                    valid_center = center_crop[np.isfinite(center_crop) & (center_crop > 0)]
                    if valid_center.size > 0:
                        center_dist = float(np.percentile(valid_center, 5))
                        if center_dist < 1.0:
                            if PRIORITY["STOP"] > PRIORITY.get(current_action, 0):
                                current_action = "STOP"
                                current_warning = f"STOP: Proximity Alert {center_dist:.2f}m CENTER"

                    visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    visual = cv2.applyColorMap(visual, cv2.COLORMAP_INFERNO)
                    visual = cv2.resize(visual, (400, 300))

                    if frame_count % 30 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                for det in detections_to_draw:
                    x1, y1, x2, y2, class_name, confidence, distance, level, action, position = det
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, max(y1 - 40, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Depth: {distance:.2f}", (x1, max(y1 - 20, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, f"{level} | {position} | {action}", (x1, min(y2 + 20, frame_h - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

                if current_action == "STOP":
                    color = (0, 0, 255)
                elif current_action in {"MOVE LEFT", "MOVE RIGHT", "KEEP LEFT", "KEEP RIGHT"}:
                    color = (0, 165, 255)
                elif current_action == "CAUTION":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                speak_navigation(current_action)

                cv2.rectangle(frame, (0, 0), (frame_w, 65), (0, 0, 0), -1)
                cv2.putText(frame, f"NAVIGATION: {current_action}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if current_warning:
                    cv2.putText(frame, current_warning, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                now = time.monotonic()
                fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now

                cv2.putText(frame, f"FPS: {fps:.1f}", (15, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Indoor Navigation System", frame)
                if visual is not None:
                    cv2.imshow("Depth Map - Visualization Only", visual)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_voice_worker()
        print("Program stopped.")

if __name__ == "__main__":
    main()