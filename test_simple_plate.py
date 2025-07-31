#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from license_plate.detector import LicensePlateDetector

# Create a simple test image with license plate
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (100, 100, 100)

# Draw vehicle
cv2.rectangle(img, (50, 50), (350, 250), (0, 0, 200), -1)

# Draw license plate
cv2.rectangle(img, (150, 200), (250, 230), (255, 255, 255), -1)
cv2.rectangle(img, (150, 200), (250, 230), (0, 0, 0), 2)
cv2.putText(img, 'ABC123', (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

cv2.imwrite('test_plate.jpg', img)
print("Created test image: test_plate.jpg")

# Test detection
detector = LicensePlateDetector()
vehicle_bbox = (50, 50, 350, 250)

results = detector.process_vehicle_for_plates(img, vehicle_bbox, "TEST_V1")
print(f"Detection results: {len(results)} plates found")

for result in results:
    print(f"Plate: {result['plate_text']} (confidence: {result['confidence']:.2f})")

# Test with camera if available
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Testing with camera...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            # Assume whole frame is vehicle for testing
            h, w = frame.shape[:2]
            results = detector.process_vehicle_for_plates(frame, (0, 0, w, h), f"CAM_V{i}")
            if results:
                print(f"Camera frame {i}: Found plates: {[r['plate_text'] for r in results]}")
    cap.release()
else:
    print("No camera available")

print("Test completed")