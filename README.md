# Vision Assist: AI Indoor Navigation Dashboard

Vision Assist is a real-time, multi-threaded AI accessibility tool designed to help visually impaired individuals navigate indoor environments safely. It utilizes a standard webcam to detect obstacles, calculate precise physical depth, and provide synchronized, priority-based spatial voice feedback. 

This project features a modern, asynchronous graphical interface built with CustomTkinter, wrapping an aggressively optimized PyTorch computer vision pipeline.

## 🚀 Key Features

* **Interactive Control Dashboard:** A sleek, dark-themed CustomTkinter UI featuring:
  * **Hardware Selector:** Dropdown menu to hot-swap between multiple connected camera inputs.
  * **Dynamic Sensitivity Slider:** Adjust the AI's "STOP" proximity threshold in real-time (0.5m to 2.5m) without restarting the app.
  * **Audio Mute Toggle:** Instantly silence the voice engine for visual-only testing.
  * **Live Telemetry HUD:** Real-time tracking of AI inference latency (ms) and system FPS.
* **Zero-Lag Multi-Threading:** Video capture, AI inference, UI rendering, and TTS audio all run on independent daemon threads communicating via thread-safe queues. This prevents the UI from freezing during heavy tensor calculations.
* **Hardware-Accelerated Inference:** Dynamically binds to NVIDIA GPUs (with automatic CPU fallback) utilizing `torch.inference_mode()`, FP16 (Half-Precision) optimizations, frame skipping, and periodic VRAM garbage collection.
* **Priority Voice Queuing:** Uses `pyttsx3` with a 1-slot priority queue. Critical danger alerts (e.g., "STOP") immediately override standard instructions (e.g., "CONTINUE"), governed by a smart cooldown system to prevent audio spam.
* **Pure-Depth Safety Net:** Employs a 5th-percentile depth cropping technique in the center frame to eliminate background bleed. Even if the YOLO model fails to classify a novel object, the system acts as a fallback and triggers a "STOP" if a physical surface breaches the proximity threshold.

## 🧠 Technology Stack

* **Frontend:** `customtkinter`, `Pillow`, `OpenCV`
* **Object Detection:** Ultralytics YOLO-World (`yolov8s-worldv2.pt`)
* **Depth Estimation:** Hugging Face Transformers (`Depth-Anything-V2-Metric-Indoor-Small`)
* **Core Processing:** `PyTorch` (CUDA), `NumPy`
* **Audio TTS:** `pyttsx3`

## ⚙️ Hardware Requirements
* **OS:** Linux (Ubuntu recommended) or Windows
* **Processing:** NVIDIA Dedicated GPU with CUDA support highly recommended (Optimized for entry-level GPUs like RTX 3050 / 4GB VRAM). Automatically falls back to CPU if no GPU is detected.
* **Camera:** Standard USB or Built-in Webcam

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/PriyaR-2006/Vision_Assist.git](https://github.com/PriyaR-2006/Vision_Assist.git)
cd Vision_Assist