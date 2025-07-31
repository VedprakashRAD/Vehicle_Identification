#!/usr/bin/env python3
"""
Quick test of vehicle detection - runs for limited frames
"""

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.working_tracker import WorkingVehicleTracker

def quick_test():
    print("🚗 Quick Vehicle Detection Test...")
    
    # Initialize tracker
    tracker = WorkingVehicleTracker(confidence_threshold=0.3)
    
    # Try vehicle test video first
    cap = cv2.VideoCapture('vehicle_test_video.mp4')
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Try camera
        if not cap.isOpened():
            print("❌ No video source available")
            return
    
    print("✅ Video source opened")
    
    frame_count = 0
    max_frames = 100  # Limit to 100 frames
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Process frame
        processed_frame, stats = tracker.process_frame_for_web(frame)
        
        frame_count += 1
        
        # Print stats every 20 frames
        if frame_count % 20 == 0:
            print(f"Frame {frame_count}: Active={stats['active_tracks']}, "
                  f"Total={stats['total_count']}, "
                  f"Cars={stats['vehicle_counts']['car']}, "
                  f"Motorcycles={stats['vehicle_counts']['motorcycle']}, "
                  f"Buses={stats['vehicle_counts']['bus']}, "
                  f"Trucks={stats['vehicle_counts']['truck']}")
    
    cap.release()
    
    # Final results
    final_stats = tracker.latest_stats
    print("\n📊 Final Results:")
    print(f"  Total Vehicles: {final_stats['total_count']}")
    print(f"  Cars: {final_stats['vehicle_counts']['car']}")
    print(f"  Motorcycles: {final_stats['vehicle_counts']['motorcycle']}")
    print(f"  Buses: {final_stats['vehicle_counts']['bus']}")
    print(f"  Trucks: {final_stats['vehicle_counts']['truck']}")
    
    if final_stats['total_count'] > 0:
        print("✅ Vehicle detection is working!")
    else:
        print("⚠️ No vehicles detected - may need real vehicle images")

if __name__ == "__main__":
    quick_test()