#!/usr/bin/env python3
"""
Script to process test video and populate the database with sample entries
"""

import cv2
import sys
import os
import logging
import time
from datetime import datetime, timedelta
from core.working_tracker import WorkingVehicleTracker
from database.manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_test_video():
    """Process test video and add sample entries to database"""
    try:
        logger.info("Initializing vehicle tracker...")
        tracker = WorkingVehicleTracker(confidence_threshold=0.3)
        
        if not tracker.model:
            logger.error("❌ Failed to load YOLO model")
            return False
            
        logger.info("✅ Vehicle tracker initialized")
        
        # Initialize database manager
        db = DatabaseManager()
        logger.info("✅ Database manager initialized")
        
        # Open test video
        video_path = "vehicle_test_video.mp4"
        if not os.path.exists(video_path):
            logger.error(f"Test video not found: {video_path}")
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open test video")
            return False
            
        logger.info(f"✅ Opened test video: {video_path}")
        
        # Process frames and simulate vehicle detections
        frame_count = 0
        detection_count = 0
        sample_plates = ["DL01AB1234", "MH02CD5678", "KA03EF9012", "TN04GH3456", "UP05IJ7890"]
        
        # Add some employee vehicles for testing
        employee_vehicles = ["DL01AB1234", "KA03EF9012"]
        for plate in employee_vehicles:
            db.add_employee_vehicle(plate, "John Doe", "Toyota", "Engineering")
            logger.info(f"Added employee vehicle: {plate}")
        
        start_time = datetime.now()
        
        while frame_count < 200:  # Process first 200 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame with tracker
            try:
                processed_frame, stats = tracker.process_frame_for_web(frame)
                
                # Simulate plate detections every 30 frames
                if frame_count % 30 == 0 and detection_count < len(sample_plates):
                    plate_text = sample_plates[detection_count]
                    is_employee = plate_text in employee_vehicles
                    vehicle_color = "Blue"
                    vehicle_brand = "Toyota" if is_employee else "Honda"
                    
                    # Simulate entry
                    entry_time = start_time + timedelta(seconds=frame_count)
                    db.log_vehicle_entry(plate_text, vehicle_color, vehicle_brand, is_employee, True)
                    logger.info(f"🚗 Vehicle ENTRY logged: {plate_text} at {entry_time}")
                    
                    # Simulate exit for some vehicles
                    if detection_count > 2:
                        exit_time = entry_time + timedelta(minutes=30)
                        db.log_vehicle_exit(plate_text)
                        logger.info(f"🚗 Vehicle EXIT logged: {plate_text} at {exit_time}")
                    
                    detection_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue
                
            # Show progress
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count} frames...")
                
            time.sleep(0.03)  # Simulate real-time processing
            
        cap.release()
        
        # Verify database entries
        logger.info("Checking database entries...")
        entries = db.get_recent_entries(20)
        logger.info(f"Database contains {len(entries)} entries")
        
        for entry in entries:
            logger.info(f"  Entry: {entry['plate_number']} - Entry: {entry['entry_time']} - Exit: {entry['exit_time']}")
        
        logger.info("✅ Test video processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in test video processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting test video processing...")
    success = process_test_video()
    if success:
        logger.info("🎉 Test video processing completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Test video processing failed!")
        sys.exit(1)