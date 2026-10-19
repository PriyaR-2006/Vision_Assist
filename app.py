import cv2
import torch
import numpy as np
import time
import threading
import queue
import pyttsx3
import gc
import customtkinter as ctk
from PIL import Image
from ultralytics import YOLOWorld
from transformers import pipeline

# ---------------- CONFIGURATION ----------------

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE = 0.15
VOICE_COOLDOWN = 4.0
PROCESS_EVERY_N_FRAMES = 3

YOLO_MODEL = "yolov8s-worldv2.pt"
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
                if phrase is None:
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
        if action == last_action and (now - last_speech_time) < VOICE_COOLDOWN:
            return
            
        last_prio = PRIORITY.get(last_action, 0)
        new_prio = PRIORITY.get(action, 0)
        
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
        
    try:
        speech_queue.put_nowait(phrase)
    except queue.Full:
        pass

# ---------------- NAVIGATION HELPERS ----------------

def get_position(center_x, width):
    if center_x < width / 3:
        return "LEFT"
    elif center_x < 2 * width / 3:
        return "CENTER"
    return "RIGHT"

def distance_level(distance):
    if distance < 0.8: return "VERY NEAR"
    elif distance < 1.5: return "NEAR"
    elif distance < 3.0: return "MEDIUM"
    return "FAR"

def get_object_depth(depth_map, x1, y1, x2, y2):
    height, width = depth_map.shape[:2]
    x1, x2 = max(0, min(int(x1), width - 1)), max(0, min(int(x2), width))
    y1, y2 = max(0, min(int(y1), height - 1)), max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1: return None

    box_w, box_h = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + 0.30 * box_w), int(x2 - 0.30 * box_w)
    cy1, cy2 = int(y1 + 0.30 * box_h), int(y2 - 0.30 * box_h)

    roi = depth_map[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size == 0: return None

    return float(np.percentile(valid, 10))

def navigation_decision(name, distance, position):
    if distance < 1.0 and position == "CENTER": return "STOP"
    if name in {"room door", "window"}:
        if distance < 1.0: return "STOP"
        if distance < 2.0: return "CAUTION"
        return "CONTINUE"
    if name == "stairs":
        if distance < 1.5: return "STOP"
        if distance < 2.5: return "CAUTION"
        return "CONTINUE"
    if name not in OBSTACLES: return "CONTINUE"

    if distance < 1.3:
        if position == "CENTER": return "STOP"
        elif position == "LEFT": return "MOVE RIGHT"
        return "MOVE LEFT"
    if distance < 2.2:
        if position == "CENTER": return "CAUTION"
        elif position == "LEFT": return "KEEP RIGHT"
        return "KEEP LEFT"
    if distance < 3.5 and position == "CENTER": return "CAUTION"

    return "CONTINUE"

def get_depth_array(result):
    predicted = result.get("predicted_depth")
    if predicted is None: raise RuntimeError("Depth model did not return predicted_depth.")
    if isinstance(predicted, torch.Tensor):
        depth = predicted.detach().float().squeeze().cpu().numpy()
    else:
        depth = np.asarray(predicted, dtype=np.float32).squeeze()
    return depth

# ---------------- UI APPLICATION ----------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VisionAssistApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vision Assist - Indoor Navigation")
        self.geometry("1000x700")
        
        # Main Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Initializing AI Models...\nPlease wait.", font=("Arial", 24))
        self.video_label.pack(expand=True)
        
        self.control_frame = ctk.CTkFrame(self, height=100)
        self.control_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        self.start_btn = ctk.CTkButton(self.control_frame, text="Start Camera", font=("Arial", 16), height=40, state="disabled", command=self.start_camera)
        self.start_btn.pack(side="left", padx=20, pady=20, expand=True)
        
        self.stop_btn = ctk.CTkButton(self.control_frame, text="Stop Camera", font=("Arial", 16), height=40, fg_color="#C62828", hover_color="#b71c1c", state="disabled", command=self.stop_camera)
        self.stop_btn.pack(side="right", padx=20, pady=20, expand=True)

        self.cap = None
        self.is_running = False
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Thread-safe queue for sending processed frames to the UI
        self.frame_queue = queue.Queue(maxsize=2)

        # Start TTS Worker
        threading.Thread(target=voice_worker, daemon=True).start()
        
        # Load Models in Background
        threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        self.detector = YOLOWorld(YOLO_MODEL)
        self.detector.set_classes(CLASSES)
        self.depth_estimator = pipeline("depth-estimation", model=DEPTH_MODEL, device=self.device)
        
        self.video_label.configure(text="Models Loaded Successfully.\nReady to start.")
        self.start_btn.configure(state="normal")

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            
            self.is_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.video_label.configure(text="")
            
            # Start the heavy processing in a background thread
            threading.Thread(target=self.ai_inference_loop, daemon=True).start()
            
            # Start the lightweight UI updater
            self.update_ui_frame()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.video_label.configure(image="")
        self.video_label.configure(text="Camera Stopped.")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        
        try:
            speech_queue.put_nowait("System stopped.")
        except queue.Full:
            pass

    def ai_inference_loop(self):
        frame_count = 0
        detections_to_draw = []
        current_action = "CONTINUE"
        current_warning = ""
        
        with torch.inference_mode():
            while self.is_running and self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    continue

                frame_count += 1
                frame_h, frame_w = frame.shape[:2]

                # Heavy AI processing
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    results = self.detector.predict(source=frame, conf=CONFIDENCE, device=self.device, verbose=False)
                    result = results[0]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_small = cv2.resize(rgb, (320, 240))
                    depth_result = self.depth_estimator(Image.fromarray(rgb_small))
                    depth_map = cv2.resize(get_depth_array(depth_result), (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

                    current_action = "CONTINUE"
                    current_warning = ""
                    detections_to_draw.clear()

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

                            if distance is None: continue

                            level = distance_level(distance)
                            action = navigation_decision(class_name, distance, position)

                            if PRIORITY.get(action, 0) > PRIORITY.get(current_action, 0):
                                current_action = action
                                current_warning = f"{action}: {class_name} {distance:.2f}m {position}"

                            detections_to_draw.append((x1, y1, x2, y2, class_name, distance, action, position))

                    # Pure-Depth Safety Net
                    center_crop = depth_map[int(frame_h*0.3):int(frame_h*0.7), int(frame_w*0.3):int(frame_w*0.7)]
                    valid_center = center_crop[np.isfinite(center_crop) & (center_crop > 0)]
                    if valid_center.size > 0:
                        center_dist = float(np.percentile(valid_center, 5))
                        if center_dist < 1.0 and PRIORITY["STOP"] > PRIORITY.get(current_action, 0):
                            current_action = "STOP"
                            current_warning = f"STOP: Proximity Alert {center_dist:.2f}m CENTER"

                    if frame_count % 30 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                # Drawing visuals (happens every frame, but based on cached AI data)
                for det in detections_to_draw:
                    x1, y1, x2, y2, class_name, distance, action, position = det
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {distance:.2f}m", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                color = (255, 0, 0) if current_action == "STOP" else (0, 255, 255) if "MOVE" in current_action or "KEEP" in current_action else (0, 255, 0)
                
                speak_navigation(current_action)

                cv2.rectangle(frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"NAV: {current_action}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if current_warning:
                    cv2.putText(frame, current_warning, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Convert to UI format and send to the UI thread
                cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv2_image)
                ctk_image = ctk.CTkImage(light_image=pil_image, size=(FRAME_WIDTH, FRAME_HEIGHT))
                
                try:
                    # Overwrite old frame if UI is slightly behind
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(ctk_image)
                except queue.Full:
                    pass

    def update_ui_frame(self):
        if not self.is_running:
            return

        try:
            # Pull the latest ready image from the background thread
            ctk_image = self.frame_queue.get_nowait()
            self.video_label.configure(image=ctk_image)
            # Important: Keep a reference to prevent garbage collection flickering
            self.video_label.image = ctk_image
        except queue.Empty:
            pass

        # Schedule next UI update very fast (15ms), as it no longer blocks AI
        self.after(15, self.update_ui_frame)

    def on_closing(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = VisionAssistApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()