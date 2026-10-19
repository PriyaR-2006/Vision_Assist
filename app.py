import cv2
import torch
import numpy as np
import time
import threading
import queue
import pyttsx3
import gc
from pathlib import Path
import customtkinter as ctk
from PIL import Image
from ultralytics import YOLOWorld
from transformers import pipeline

# ---------------- CONFIGURATION & GLOBALS ----------------

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE = 0.15
VOICE_COOLDOWN = 4.0
PROCESS_EVERY_N_FRAMES = 3

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

# Dynamic UI States
AUDIO_MUTED = False
STOP_THRESH = 1.0

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
                if not AUDIO_MUTED:
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
        if voice_shutdown or AUDIO_MUTED:
            return
        if action == last_action and (now - last_speech_time) < VOICE_COOLDOWN:
            return
            
        last_prio = PRIORITY.get(last_action, 0)
        new_prio = PRIORITY.get(action, 0)
        
        if new_prio < last_prio and (now - last_speech_time) < VOICE_COOLDOWN:
            return

        phrase = PHRASES.get(action)
        if phrase is None: return
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
    if center_x < width / 3: return "LEFT"
    elif center_x < 2 * width / 3: return "CENTER"
    return "RIGHT"

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
    caution_thresh = STOP_THRESH + 1.0

    if distance < STOP_THRESH and position == "CENTER": return "STOP"
    
    if name in {"room door", "window", "stairs"}:
        if distance < STOP_THRESH + 0.2: return "STOP"
        if distance < caution_thresh: return "CAUTION"
        return "CONTINUE"
        
    if name not in OBSTACLES: return "CONTINUE"

    if distance < STOP_THRESH + 0.3:
        if position == "CENTER": return "STOP"
        elif position == "LEFT": return "MOVE RIGHT"
        return "MOVE LEFT"
        
    if distance < caution_thresh:
        if position == "CENTER": return "CAUTION"
        elif position == "LEFT": return "KEEP RIGHT"
        return "KEEP LEFT"

    return "CONTINUE"

def get_depth_array(result):
    predicted = result.get("predicted_depth")
    if isinstance(predicted, torch.Tensor):
        return predicted.detach().float().squeeze().cpu().numpy()
    return np.asarray(predicted, dtype=np.float32).squeeze()

# ---------------- UI APPLICATION ----------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VisionAssistApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vision Assist - Professional Dashboard")
        self.geometry("1100x650")
        
        # Grid Layout (Video on left, Sidebar on right)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)
        
        # --- LEFT: VIDEO FRAME ---
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="Initializing AI Models...\nPlease wait.", font=("Arial", 24))
        self.video_label.pack(expand=True)
        
        # --- RIGHT: SIDEBAR (CONTROLS & HUD) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280)
        self.sidebar_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Vision Assist UI", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(pady=(20, 30))
        
        # 1. Camera Dropdown
        self.cam_label = ctk.CTkLabel(self.sidebar_frame, text="Select Camera Input:")
        self.cam_label.pack(anchor="w", padx=20)
        self.camera_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["Camera 0", "Camera 1", "Camera 2"])
        self.camera_option.pack(padx=20, pady=(0, 20), fill="x")
        
        # 2. Audio Mute Toggle
        self.mute_switch = ctk.CTkSwitch(self.sidebar_frame, text="Mute Voice Guidance", command=self.toggle_mute)
        self.mute_switch.pack(padx=20, pady=10, anchor="w")
        
        # 3. Dynamic Distance Slider
        self.slider_label = ctk.CTkLabel(self.sidebar_frame, text=f"STOP Threshold: {STOP_THRESH:.1f}m")
        self.slider_label.pack(anchor="w", padx=20, pady=(20, 5))
        self.dist_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.5, to=2.5, command=self.update_threshold)
        self.dist_slider.set(STOP_THRESH)
        self.dist_slider.pack(padx=20, pady=(0, 30), fill="x")
        
        # Start/Stop Buttons
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Start Camera", height=40, state="disabled", command=self.start_camera)
        self.start_btn.pack(padx=20, pady=10, fill="x")
        
        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Stop Camera", height=40, fg_color="#C62828", hover_color="#b71c1c", state="disabled", command=self.stop_camera)
        self.stop_btn.pack(padx=20, pady=10, fill="x")

        # 4. Performance HUD
        self.hud_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.hud_frame.pack(side="bottom", pady=20, fill="x")
        self.hud_label = ctk.CTkLabel(self.hud_frame, text="FPS: 0.0\nAI Latency: 0 ms", font=ctk.CTkFont(size=14), justify="left")
        self.hud_label.pack()

        # Threading/App State
        self.cap = None
        self.is_running = False
        self.models_ready = False
        self._camera_thread = None
        self._camera_stop_event = None
        self._closing = False
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.ui_event_queue = queue.Queue()
        
        # Live Stats variables
        self.current_fps = 0.0
        self.current_ai_time = 0.0

        # Boot Backend
        threading.Thread(target=voice_worker, daemon=True).start()
        threading.Thread(target=self.load_models, daemon=True).start()
        self.after(50, self._process_ui_events)

    def toggle_mute(self):
        global AUDIO_MUTED
        AUDIO_MUTED = self.mute_switch.get() == 1

    def update_threshold(self, value):
        global STOP_THRESH
        STOP_THRESH = float(value)
        self.slider_label.configure(text=f"STOP Threshold: {STOP_THRESH:.1f}m")

    def load_models(self):
        try:
            self.detector = YOLOWorld(str(YOLO_MODEL))
            self.detector.set_classes(CLASSES)
            if self.device == 0:
                self.detector.model.half()
            
            dtype = torch.float16 if self.device == 0 else torch.float32
            self.depth_estimator = pipeline("depth-estimation", model=DEPTH_MODEL, device=self.device, dtype=dtype)
        except Exception as exc:
            self.ui_event_queue.put(("model_load_error", str(exc)))
            return
        self.ui_event_queue.put(("models_loaded", None))

    def _process_ui_events(self):
        if self._closing: return
        try:
            while True:
                event, payload = self.ui_event_queue.get_nowait()
                if event == "models_loaded":
                    self.models_ready = True
                    self.video_label.configure(text="Models Loaded Successfully.\nReady to start.")
                    self.start_btn.configure(state="normal")
                elif event == "camera_finished":
                    if self._camera_stop_event is payload and self.is_running:
                        self.stop_camera()
        except queue.Empty:
            pass
        self.after(50, self._process_ui_events)

    def start_camera(self):
        if self._closing or not self.models_ready or self.is_running: return
        
        cam_idx = int(self.camera_option.get().split(" ")[1])
        
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()

        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            self.video_label.configure(text="Could not open selected camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.cap = cap
        self._camera_stop_event = threading.Event()
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.video_label.configure(text="")

        self._camera_thread = threading.Thread(target=self._run_ai_inference_loop, args=(cap, self._camera_stop_event), daemon=True)
        self._camera_thread.start()
        self.update_ui_frame()

    def stop_camera(self):
        self.is_running = False
        if self._camera_stop_event: self._camera_stop_event.set()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image="")
        self.video_label.configure(text="Camera Stopped.")
        self.stop_btn.configure(state="disabled")
        if not self._closing and self.models_ready:
            self.start_btn.configure(state="normal")
        try: speech_queue.put_nowait("System stopped.")
        except queue.Full: pass

    def _run_ai_inference_loop(self, cap, stop_event):
        frame_count = 0
        detections_to_draw = []
        current_action = "CONTINUE"
        current_warning = ""
        
        time_prev = time.monotonic()

        with torch.inference_mode():
            while not stop_event.is_set() and cap.isOpened():
                success, frame = cap.read()
                if not success: break

                frame_count += 1
                frame_h, frame_w = frame.shape[:2]

                time_now = time.monotonic()
                self.current_fps = 1.0 / max(time_now - time_prev, 1e-6)
                time_prev = time_now

                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    ai_start_time = time.monotonic()
                    
                    results = self.detector.predict(source=frame, conf=CONFIDENCE, device=self.device, imgsz=320, verbose=False)
                    rgb_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 240))
                    depth_result = self.depth_estimator(Image.fromarray(rgb_small))
                    depth_map = cv2.resize(get_depth_array(depth_result), (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

                    current_action = "CONTINUE"
                    current_warning = ""
                    detections_to_draw.clear()

                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
                            class_id = int(box.cls[0].detach().cpu())
                            class_name = results[0].names[class_id]

                            position = get_position((x1 + x2) // 2, frame_w)
                            distance = get_object_depth(depth_map, x1, y1, x2, y2)

                            if distance is None: continue

                            action = navigation_decision(class_name, distance, position)
                            if PRIORITY.get(action, 0) > PRIORITY.get(current_action, 0):
                                current_action = action
                                current_warning = f"{action}: {class_name} {distance:.2f}m {position}"

                            detections_to_draw.append((x1, y1, x2, y2, class_name, distance, action, position))

                    center_crop = depth_map[int(frame_h*0.3):int(frame_h*0.7), int(frame_w*0.3):int(frame_w*0.7)]
                    valid_center = center_crop[np.isfinite(center_crop) & (center_crop > 0)]
                    if valid_center.size > 0:
                        center_dist = float(np.percentile(valid_center, 5))
                        if center_dist < STOP_THRESH and PRIORITY["STOP"] > PRIORITY.get(current_action, 0):
                            current_action = "STOP"
                            current_warning = f"STOP: Proximity Alert {center_dist:.2f}m CENTER"
                            
                    self.current_ai_time = (time.monotonic() - ai_start_time) * 1000

                    if frame_count % 30 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                # --- VISUAL RENDERING ---
                for det in detections_to_draw:
                    x1, y1, x2, y2, class_name, distance, _, _ = det
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {distance:.2f}m", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                color = (255, 0, 0) if current_action == "STOP" else (0, 255, 255) if "MOVE" in current_action or "KEEP" in current_action else (0, 255, 0)
                
                # Top Black Banner
                cv2.rectangle(frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"NAV: {current_action}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Warning Text Below Banner
                if current_warning:
                    cv2.putText(frame, current_warning, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                speak_navigation(current_action)

                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try: self.frame_queue.put_nowait(pil_image)
                except queue.Full: pass

        self.ui_event_queue.put(("camera_finished", stop_event))

    def update_ui_frame(self):
        if not self.is_running or self._closing: return
        try:
            pil_image = self.frame_queue.get_nowait()
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(FRAME_WIDTH, FRAME_HEIGHT))
            self.video_label.configure(image=ctk_image)
            self.video_label.image = ctk_image
        except queue.Empty: pass

        self.hud_label.configure(text=f"FPS: {self.current_fps:.1f}\nAI Latency: {self.current_ai_time:.0f} ms")
        self.after(15, self.update_ui_frame)

    def on_closing(self):
        if self._closing: return
        self._closing = True
        self.stop_camera()
        stop_voice_worker()
        self.destroy()

if __name__ == "__main__":
    app = VisionAssistApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()