#!/usr/bin/env python3
"""
Test YOLOv5 vehicle detection
"""

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.working_tracker import WorkingVehicleTracker

def test_detection():
    print("🚗 Testing Vehicle Detection...")
    
    # Initialize tracker
    tracker = WorkingVehicleTracker(confidence_threshold=0.5)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available, using test video")
        cap = cv2.VideoCapture('test_video.mp4')
    
    if not cap.isOpened():
        print("❌ No video source available")
        return
    
    print("✅ Video source opened")
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera error")
            break
        
        # Process frame
        processed_frame, stats = tracker.process_frame_for_web(frame)
        
        # Display results
        cv2.imshow('Vehicle Detection Test', processed_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print stats every 30 frames
            print(f"Frame {frame_count}: {stats}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'detection_result_{frame_count}.jpg', processed_frame)
            print(f"Saved frame {frame_count}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Test completed")

if __name__ == "__main__":
    test_detection()