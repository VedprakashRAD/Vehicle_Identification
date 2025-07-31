#!/usr/bin/env python3
"""
Test all vehicle types detection with license plate recognition
"""

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.working_tracker import WorkingVehicleTracker

def test_all_vehicles():
    print("🚗 Testing All Vehicle Types Detection...")
    
    # Initialize tracker with lower confidence for better detection
    tracker = WorkingVehicleTracker(confidence_threshold=0.3)
    
    # Try camera first
    cap = cv2.VideoCapture(0)
    source_name = "Camera"
    
    if not cap.isOpened():
        print("📹 Camera not available, using test video...")
        cap = cv2.VideoCapture('vehicle_test_video.mp4')
        source_name = "Test Video"
    
    if not cap.isOpened():
        print("❌ No video source available")
        return
    
    print(f"✅ {source_name} opened successfully")
    print("🎯 Vehicle Detection Colors:")
    print("  - Cars: Green boxes")
    print("  - Motorcycles: Blue boxes") 
    print("  - Buses: Yellow boxes")
    print("  - Trucks: Magenta boxes")
    print("  - License Plates: Yellow boxes with text")
    print("Press 'q' to quit, 'r' to reset counts")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if source_name == "Test Video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # Process frame
        processed_frame, stats = tracker.process_frame_for_web(frame)
        
        # Display frame
        cv2.imshow('All Vehicle Types Detection', processed_frame)
        
        # Print stats every 60 frames
        if frame_count % 60 == 0:
            print(f"📊 Frame {frame_count}:")
            print(f"  Cars: {stats['vehicle_counts']['car']}")
            print(f"  Motorcycles: {stats['vehicle_counts']['motorcycle']}")
            print(f"  Buses: {stats['vehicle_counts']['bus']}")
            print(f"  Trucks: {stats['vehicle_counts']['truck']}")
            print(f"  License Plates: {len(stats.get('license_plates', []))}")
            print(f"  Active Vehicles: {stats['active_tracks']}")
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_counts()
            print("🔄 Counts reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    final_stats = tracker.latest_stats
    print("\n📊 Final Detection Results:")
    print(f"  Total Vehicles: {final_stats['total_count']}")
    print(f"  Cars: {final_stats['vehicle_counts']['car']}")
    print(f"  Motorcycles: {final_stats['vehicle_counts']['motorcycle']}")
    print(f"  Buses: {final_stats['vehicle_counts']['bus']}")
    print(f"  Trucks: {final_stats['vehicle_counts']['truck']}")
    print(f"  License Plates Detected: {len(final_stats.get('license_plates', []))}")
    
    # Show license plate details
    plates = final_stats.get('license_plates', [])
    if plates:
        print("\n🔢 License Plates Detected:")
        for plate in plates[-5:]:  # Show last 5
            print(f"  - {plate['plate_text']} (Vehicle: {plate['vehicle_id']}, Confidence: {plate['confidence']:.2f})")
    
    print("🏁 Test completed")

if __name__ == "__main__":
    test_all_vehicles()