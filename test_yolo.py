#!/usr/bin/env python3
"""
Test ultralytics YOLO implementation
"""

from ultralytics import YOLO
import cv2
import numpy as np

def test_yolo():
    print("🔧 Testing Ultralytics YOLO...")
    
    try:
        # Load model
        print("📦 Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        print("✅ Model loaded successfully")
        
        # Create test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (100, 150, 200)
        
        # Add some shapes
        cv2.rectangle(test_img, (100, 100), (200, 180), (255, 255, 255), -1)
        cv2.rectangle(test_img, (300, 200), (450, 300), (200, 200, 200), -1)
        
        # Run inference
        print("🎯 Running inference...")
        results = model(test_img, conf=0.5)
        
        print(f"📊 Results: {len(results)} result objects")
        
        # Vehicle classes
        vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"🚗 Found {len(boxes)} detections")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    
                    print(f"  - Class {class_id} ({class_name}): {confidence:.3f}")
                    
                    if class_id in vehicle_classes:
                        print(f"    🎯 Vehicle detected: {vehicle_classes[class_id]}")
            else:
                print("📭 No detections found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_yolo()
    if success:
        print("✅ YOLO test completed!")
    else:
        print("❌ YOLO test failed!")