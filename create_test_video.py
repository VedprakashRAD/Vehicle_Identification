#!/usr/bin/env python3
"""
Create a test video with moving rectangles to simulate vehicles
"""

import cv2
import numpy as np

def create_test_video():
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    print(f"🎬 Creating test video: {width}x{height}, {fps}fps, {duration}s")
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 60, 80)  # Dark blue background
        
        # Add moving rectangles (simulating vehicles)
        x1 = int((frame_num * 2) % (width + 100)) - 50
        x2 = int((frame_num * 1.5 + 200) % (width + 100)) - 50
        
        # Vehicle 1 (Car)
        if 0 <= x1 <= width - 80:
            cv2.rectangle(frame, (x1, 200), (x1 + 80, 250), (0, 255, 0), -1)
            cv2.putText(frame, 'CAR', (x1 + 10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Vehicle 2 (Truck)
        if 0 <= x2 <= width - 100:
            cv2.rectangle(frame, (x2, 300), (x2 + 100, 360), (0, 0, 255), -1)
            cv2.putText(frame, 'TRUCK', (x2 + 10, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f'Test Video - Frame {frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'AI Vehicle Detection Demo', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"📹 Progress: {frame_num}/{total_frames} frames")
    
    out.release()
    print("✅ Test video created: test_video.mp4")

if __name__ == "__main__":
    create_test_video()