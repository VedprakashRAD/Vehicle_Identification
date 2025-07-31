#!/usr/bin/env python3
"""
Quick test to check if YOLOv5 model loads and works
"""

import torch
import cv2
import numpy as np
from pathlib import Path

def test_model():
    print("🔧 Testing YOLOv5 Model Loading...")
    
    try:
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Device: {device}")
        
        # Load YOLOv5s pretrained model
        print("📦 Loading YOLOv5s pretrained model")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("✅ Pretrained model loaded successfully")
        
        model.to(device)
        model.conf = 0.5
        
        # Test with a simple image
        print("🖼️ Testing with sample image...")
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (100, 150, 200)  # Fill with color
        
        # Add some rectangles to simulate vehicles
        cv2.rectangle(test_img, (100, 100), (200, 180), (255, 255, 255), -1)
        cv2.rectangle(test_img, (300, 200), (450, 300), (200, 200, 200), -1)
        
        # Run inference
        results = model(test_img)
        detections = results.pandas().xyxy[0]
        
        print(f"🎯 Detections found: {len(detections)}")
        if len(detections) > 0:
            print("Detection details:")
            for i, detection in detections.iterrows():
                class_name = detection['name']
                confidence = detection['confidence']
                print(f"  - {class_name}: {confidence:.3f}")
        
        # Vehicle classes in COCO
        vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        print(f"🚗 Vehicle classes to detect: {list(vehicle_classes.values())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("✅ Model test completed successfully!")
    else:
        print("❌ Model test failed!")