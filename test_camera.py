#!/usr/bin/env python3
"""
Simple camera test script to check if camera is working
"""

import cv2
import sys

def test_camera():
    print("🎥 Testing camera access...")
    
    # Test different camera indices
    for camera_idx in range(5):  # Test indices 0-4
        print(f"\n📹 Testing camera index {camera_idx}...")
        
        cap = cv2.VideoCapture(camera_idx)
        
        if cap.isOpened():
            print(f"✅ Camera {camera_idx} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {camera_idx} can read frames - Resolution: {frame.shape}")
                print(f"🎯 Camera {camera_idx} is WORKING!")
                
                # Show frame for 2 seconds
                cv2.imshow(f'Camera {camera_idx} Test', frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
            else:
                print(f"❌ Camera {camera_idx} opened but cannot read frames")
        else:
            print(f"❌ Camera {camera_idx} failed to open")
        
        cap.release()
    
    print("\n🔍 Camera test completed!")

if __name__ == "__main__":
    test_camera()