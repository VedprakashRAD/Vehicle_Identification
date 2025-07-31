#!/usr/bin/env python3
"""
Test license plate detection and recognition
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from license_plate.detector import LicensePlateDetector

def create_test_license_plate():
    """Create a synthetic license plate image for testing"""
    # Create a license plate-like image
    plate_img = np.ones((60, 200, 3), dtype=np.uint8) * 255  # White background
    
    # Add border
    cv2.rectangle(plate_img, (5, 5), (195, 55), (0, 0, 0), 2)
    
    # Add text
    cv2.putText(plate_img, 'ABC123', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    return plate_img

def test_license_plate_detection():
    print("🔍 Testing License Plate Detection and Recognition...")
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    # Test 1: Synthetic license plate
    print("\n📝 Test 1: Synthetic License Plate")
    test_plate = create_test_license_plate()
    
    # Save test image
    cv2.imwrite('test_license_plate.jpg', test_plate)
    print("💾 Saved test license plate image: test_license_plate.jpg")
    
    # Test OCR on the synthetic plate
    result = detector.recognize_text(test_plate)
    if result:
        print(f"✅ OCR Result: '{result['text']}' (confidence: {result['confidence']:.2f})")
    else:
        print("❌ No text recognized from synthetic plate")
    
    # Test 2: Camera/Video detection
    print("\n📹 Test 2: Live Detection")
    
    # Try camera first, then test video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture('vehicle_test_video.mp4')
        if not cap.isOpened():
            print("❌ No video source available")
            return
    
    print("✅ Video source opened")
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    max_frames = 200  # Limit frames for testing
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Detect license plates in the full frame
        plates = detector.detect_license_plates(frame)
        
        if plates:
            print(f"🎯 Frame {frame_count}: Found {len(plates)} potential license plates")
            
            for i, plate in enumerate(plates):
                # Try to recognize text
                result = detector.recognize_text(plate['image'])
                if result:
                    print(f"  Plate {i+1}: '{result['text']}' (confidence: {result['confidence']:.2f})")
                    
                    # Draw detection on frame
                    x1, y1, x2, y2 = plate['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, result['text'], (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('License Plate Detection Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'license_test_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"💾 Saved: {filename}")
        
        frame_count += 1
        
        # Print progress every 50 frames
        if frame_count % 50 == 0:
            print(f"📊 Processed {frame_count} frames...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Show recent detections
    recent = detector.get_recent_detections(10)
    if recent:
        print(f"\n📋 Recent License Plate Detections ({len(recent)}):")
        for detection in recent:
            print(f"  - {detection['plate_text']} (confidence: {detection['confidence']:.2f})")
    else:
        print("\n📭 No license plates detected")
    
    print("🏁 License plate test completed")

if __name__ == "__main__":
    test_license_plate_detection()