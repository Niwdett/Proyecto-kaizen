import cv2
import torch
from ultralytics import YOLO
import onnxruntime as ort
import sqlite3

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("ONNX Runtime:", ort.__version__)

model = YOLO("yolov8n.pt")  # Descarga el modelo base
print("YOLOv8n cargado correctamente ✅")