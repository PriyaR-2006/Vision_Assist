# Vision Assist: Indoor Navigation Assistant

Vision Assist is an assistive computer vision system designed to help visually impaired individuals navigate indoor environments. It combines zero-shot object detection with metric monocular depth estimation and audio cues.

## Features (Current Baseline)
- **Object Detection:** Powered by `YOLO-World` (`yolov8s-worldv2.pt`) for real-time promptable detection of common indoor obstacles (doors, chairs, stairs, people, etc.).
- **Depth Estimation:** Uses `Depth-Anything-V2-Metric-Indoor-Small-hf` via Hugging Face Transformers to estimate real-world distance.
- **Audio Feedback:** Real-time spatial instructions ("Move Left", "Caution", "Stop") driven by `pyttsx3`.

## System Requirements & Testbed
- **Hardware:** ASUS Laptop with NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **Frameworks:** Python 3.10+, PyTorch (CUDA enabled), OpenCV, Ultralytics, Transformers

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Praveena-2907/Vission_Assist.git](https://github.com/Praveena-2907/Vission_Assist.git)
   cd Vission_Assist