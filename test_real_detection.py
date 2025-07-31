#!/usr/bin/env python3
"""
Test vehicle detection with real camera feed
"""

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.working_tracker import WorkingVehicleTracker

def test_real_detection():
    print("🚗 Testing Real Vehicle Detection...")
    
    # Initialize tracker
    tracker = WorkingVehicleTracker(confidence_threshold=0.3)  # Lower threshold for better detection
    
    # Try camera first, then test video
    cap = cv2.VideoCapture(0)
    source_name = "Camera"
    
    if not cap.isOpened():
        print("📹 Camera not available, trying vehicle test video...")
        cap = cv2.VideoCapture('vehicle_test_video.mp4')
        source_name = "Vehicle Test Video"
        
        if not cap.isOpened():
            print("📹 Vehicle test video not found, trying basic test video...")
            cap = cv2.VideoCapture('test_video.mp4')
            source_name = "Basic Test Video"
    
    if not cap.isOpened():
        print("❌ No video source available")
        return
    
    print(f"✅ {source_name} opened successfully")
    print("🎯 Detection Info:")
    print("  - Cars: Green boxes")
    print("  - Motorcycles: Blue boxes") 
    print("  - Buses: Yellow boxes")
    print("  - Trucks: Magenta boxes")
    print("Press 'q' to quit, 'r' to reset counts, 's' to save frame")
    
    frame_count = 0
    last_stats = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if source_name == "Test Video":
                # Loop the test video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("❌ Failed to read frame")
                break
        
        # Process frame with detection
        processed_frame, stats = tracker.process_frame_for_web(frame)
        
        # Add detection info overlay
        info_text = [
            f"Frame: {frame_count}",
            f"Source: {source_name}",
            f"Active Detections: {stats['active_tracks']}",
            f"Total Vehicles: {stats['total_count']}",
            f"Cars: {stats['vehicle_counts']['car']}",
            f"Motorcycles: {stats['vehicle_counts']['motorcycle']}",
            f"Buses: {stats['vehicle_counts']['bus']}",
            f"Trucks: {stats['vehicle_counts']['truck']}"
        ]
        
        # Draw info panel
        y_offset = 150
        for i, text in enumerate(info_text):
            cv2.putText(processed_frame, text, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Vehicle Detection Test', processed_frame)
        
        # Print stats when they change
        if stats != last_stats:
            print(f"🎯 Detection Update - Active: {stats['active_tracks']}, "
                  f"Total: {stats['total_count']}, "
                  f"Cars: {stats['vehicle_counts']['car']}, "
                  f"Motorcycles: {stats['vehicle_counts']['motorcycle']}, "
                  f"Buses: {stats['vehicle_counts']['bus']}, "
                  f"Trucks: {stats['vehicle_counts']['truck']}")
            last_stats = stats.copy()
        
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_counts()
            print("🔄 Counts reset")
        elif key == ord('s'):
            filename = f'detection_result_{frame_count}.jpg'
            cv2.imwrite(filename, processed_frame)
            print(f"💾 Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    final_stats = tracker.latest_stats
    print("\n📊 Final Detection Results:")
    print(f"  Total Vehicles Detected: {final_stats['total_count']}")
    print(f"  Cars: {final_stats['vehicle_counts']['car']}")
    print(f"  Motorcycles: {final_stats['vehicle_counts']['motorcycle']}")
    print(f"  Buses: {final_stats['vehicle_counts']['bus']}")
    print(f"  Trucks: {final_stats['vehicle_counts']['truck']}")
    print("🏁 Test completed")

if __name__ == "__main__":
    test_real_detection()