#!/usr/bin/env python3
"""
yolo26n.pt → TensorRT engine 변환
실행: cd scripts/yolo_detector && python3 export_trt.py
결과: weights/yolo26n.engine
"""
import os
from ultralytics import YOLO

pt_path = os.path.join(os.path.dirname(__file__), "weights/yolo26n.pt")
print(f"Loading: {pt_path}")

model = YOLO(pt_path)
print("Exporting to TensorRT FP16 (imgsz=352)...")
engine_path = model.export(
    format="engine",
    half=True,
    device=0,
    imgsz=352,
    simplify=True,
)
print(f"Done: {engine_path}")
print("Move engine file to weights/ if needed.")
