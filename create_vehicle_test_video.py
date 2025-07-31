#!/usr/bin/env python3
"""
Create a test video with different vehicle types for better detection testing
"""

import cv2
import numpy as np

def create_vehicle_test_video():
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 15  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('vehicle_test_video.mp4', fourcc, fps, (width, height))
    
    print(f"🎬 Creating vehicle test video: {width}x{height}, {fps}fps, {duration}s")
    
    for frame_num in range(total_frames):
        # Create background (road-like)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (60, 80, 60)  # Dark green background
        
        # Draw road
        cv2.rectangle(frame, (0, height//2), (width, height), (40, 40, 40), -1)
        
        # Road markings
        for x in range(0, width, 60):
            cv2.rectangle(frame, (x, height//2 + 50), (x + 30, height//2 + 60), (255, 255, 255), -1)
        
        # Moving vehicles with different speeds and positions
        time_factor = frame_num / fps
        
        # Car 1 (moving left to right)
        car1_x = int((time_factor * 80) % (width + 120)) - 60
        if 0 <= car1_x <= width - 80:
            # Car body
            cv2.rectangle(frame, (car1_x, 200), (car1_x + 80, 240), (0, 0, 255), -1)  # Red car
            # Windows
            cv2.rectangle(frame, (car1_x + 10, 205), (car1_x + 70, 220), (100, 150, 200), -1)
            # Wheels
            cv2.circle(frame, (car1_x + 15, 240), 8, (0, 0, 0), -1)
            cv2.circle(frame, (car1_x + 65, 240), 8, (0, 0, 0), -1)
        
        # Truck (moving right to left, slower)
        truck_x = width - int((time_factor * 50) % (width + 150))
        if 0 <= truck_x <= width - 120:
            # Truck body
            cv2.rectangle(frame, (truck_x, 180), (truck_x + 120, 240), (255, 0, 0), -1)  # Blue truck
            # Cab
            cv2.rectangle(frame, (truck_x + 90, 160), (truck_x + 120, 200), (200, 0, 0), -1)
            # Wheels
            cv2.circle(frame, (truck_x + 20, 240), 10, (0, 0, 0), -1)
            cv2.circle(frame, (truck_x + 50, 240), 10, (0, 0, 0), -1)
            cv2.circle(frame, (truck_x + 100, 240), 10, (0, 0, 0), -1)
        
        # Motorcycle (fast, small)
        bike_x = int((time_factor * 120) % (width + 60)) - 30
        if 0 <= bike_x <= width - 40:
            # Bike body
            cv2.rectangle(frame, (bike_x, 220), (bike_x + 40, 235), (0, 255, 0), -1)  # Green bike
            # Wheels
            cv2.circle(frame, (bike_x + 8, 235), 6, (0, 0, 0), -1)
            cv2.circle(frame, (bike_x + 32, 235), 6, (0, 0, 0), -1)
            # Rider
            cv2.circle(frame, (bike_x + 20, 210), 8, (255, 200, 150), -1)
        
        # Bus (large, slow)
        bus_x = int((time_factor * 30) % (width + 200)) - 100
        if 0 <= bus_x <= width - 150:
            # Bus body
            cv2.rectangle(frame, (bus_x, 160), (bus_x + 150, 240), (255, 255, 0), -1)  # Yellow bus
            # Windows
            for i in range(5):
                cv2.rectangle(frame, (bus_x + 10 + i * 25, 170), (bus_x + 30 + i * 25, 190), (150, 200, 255), -1)
            # Wheels
            cv2.circle(frame, (bus_x + 25, 240), 12, (0, 0, 0), -1)
            cv2.circle(frame, (bus_x + 125, 240), 12, (0, 0, 0), -1)
        
        # Car 2 (different lane)
        car2_x = int((time_factor * 60 + 200) % (width + 100)) - 50
        if 0 <= car2_x <= width - 70:
            # Car body
            cv2.rectangle(frame, (car2_x, 280), (car2_x + 70, 315), (255, 0, 255), -1)  # Magenta car
            # Windows
            cv2.rectangle(frame, (car2_x + 10, 285), (car2_x + 60, 300), (100, 150, 200), -1)
            # Wheels
            cv2.circle(frame, (car2_x + 12, 315), 8, (0, 0, 0), -1)
            cv2.circle(frame, (car2_x + 58, 315), 8, (0, 0, 0), -1)
        
        # Add frame info
        cv2.putText(frame, f'Vehicle Test Video - Frame {frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, 'Multiple Vehicle Types for Detection', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Vehicle labels
        cv2.putText(frame, 'Red=Car, Blue=Truck, Green=Motorcycle, Yellow=Bus, Magenta=Car', 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        out.write(frame)
        
        if frame_num % 60 == 0:
            print(f"📹 Progress: {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
    
    out.release()
    print("✅ Vehicle test video created: vehicle_test_video.mp4")

if __name__ == "__main__":
    create_vehicle_test_video()