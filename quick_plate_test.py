#!/usr/bin/env python3
"""
Quick test for license plate detection
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from license_plate.detector import LicensePlateDetector

def create_test_image():
    """Create a test image with a vehicle and license plate"""
    # Create a simple test image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (100, 150, 100)  # Green background
    
    # Draw a simple car shape
    cv2.rectangle(img, (150, 100), (450, 300), (50, 50, 200), -1)  # Car body
    cv2.rectangle(img, (120, 200), (180, 250), (0, 0, 0), -1)      # Wheel
    cv2.rectangle(img, (420, 200), (480, 250), (0, 0, 0), -1)      # Wheel
    
    # Draw license plate area
    cv2.rectangle(img, (250, 280), (350, 320), (255, 255, 255), -1)  # White plate
    cv2.rectangle(img, (250, 280), (350, 320), (0, 0, 0), 2)        # Black border
    
    # Add license plate text
    cv2.putText(img, 'ABC123', (260, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img

def main():
    print("🔍 Quick License Plate Detection Test")
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    # Create test image
    test_img = create_test_image()
    cv2.imwrite('test_car_image.jpg', test_img)
    print("💾 Created test image: test_car_image.jpg")
    
    # Test detection
    print("\n📋 Testing license plate detection...")
    
    # Define vehicle bounding box (the car area)
    vehicle_bbox = (150, 100, 450, 300)
    
    # Detect plates
    plates = detector.detect_license_plates(test_img, vehicle_bbox)
    print(f"🎯 Found {len(plates)} potential license plates")
    
    if plates:
        for i, plate in enumerate(plates):
            print(f"\nPlate {i+1}:")
            print(f"  Bbox: {plate['bbox']}")
            print(f"  Confidence: {plate['confidence']:.2f}")
            
            # Try OCR
            result = detector.recognize_text(plate['image'])
            if result:
                print(f"  Text: '{result['text']}' (confidence: {result['confidence']:.2f})")
            else:
                print("  Text: No text recognized")
            
            # Save plate image
            cv2.imwrite(f'detected_plate_{i+1}.jpg', plate['image'])
    else:
        print("❌ No license plates detected")
        
        # Try fallback detection
        print("\n🔄 Trying fallback detection...")
        fallback_plates = detector.fallback_plate_detection(test_img, vehicle_bbox)
        print(f"🎯 Fallback found {len(fallback_plates)} potential plates")
        
        for i, plate in enumerate(fallback_plates):
            result = detector.recognize_text(plate['image'])
            if result:
                print(f"  Fallback Plate {i+1}: '{result['text']}'")
    
    # Test with real camera/video if available
    print("\n📹 Testing with camera/video...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture('vehicle_test_video.mp4')
    
    if cap.isOpened():
        print("✅ Video source opened - processing 10 frames...")
        for frame_num in range(10):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Try to detect plates in the full frame
            plates = detector.detect_license_plates(frame)
            if plates:
                print(f"Frame {frame_num}: Found {len(plates)} plates")
                for plate in plates:
                    result = detector.recognize_text(plate['image'])
                    if result and result['text']:
                        print(f"  Detected: {result['text']}")
        cap.release()
    else:
        print("❌ No video source available")
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    main()