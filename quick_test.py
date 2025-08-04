#!/usr/bin/env python3
"""
Quick test script for vehicle detection system
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.working_tracker import WorkingVehicleTracker
from core.camera_manager import CameraManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test of vehicle detection system"""
    logger.info("🚗 Starting Quick Vehicle Detection Test...")
    
    # Initialize components
    tracker = WorkingVehicleTracker(confidence_threshold=0.3)
    camera_manager = CameraManager()
    
    # Initialize camera
    if not camera_manager.initialize_camera():
        logger.error("❌ Failed to initialize camera or video source")
        return
    
    logger.info("✅ Camera/video source initialized")
    logger.info(f"Source info: {camera_manager.get_source_info()}")
    
    frame_count = 0
    max_frames = 100  # Limit test to 100 frames
    
    try:
        while frame_count < max_frames:
            ret, frame = camera_manager.read_frame()
            if not ret:
                logger.warning("Failed to read frame")
                continue
            
            # Process frame
            processed_frame, stats = tracker.process_frame_for_web(frame)
            frame_count += 1
            
            # Print stats every 20 frames
            if frame_count % 20 == 0:
                logger.info(f"Frame {frame_count}: Active={stats['active_tracks']}, "
                          f"Total={stats['total_count']}, "
                          f"Cars={stats['vehicle_counts']['car']}, "
                          f"Motorcycles={stats['vehicle_counts']['motorcycle']}, "
                          f"Buses={stats['vehicle_counts']['bus']}, "
                          f"Trucks={stats['vehicle_counts']['truck']}")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    finally:
        camera_manager.release()
    
    # Final results
    final_stats = tracker.latest_stats
    logger.info("\n📊 Final Results:")
    logger.info(f"  Total Vehicles: {final_stats['total_count']}")
    logger.info(f"  Cars: {final_stats['vehicle_counts']['car']}")
    logger.info(f"  Motorcycles: {final_stats['vehicle_counts']['motorcycle']}")
    logger.info(f"  Buses: {final_stats['vehicle_counts']['bus']}")
    logger.info(f"  Trucks: {final_stats['vehicle_counts']['truck']}")
    
    if final_stats['total_count'] > 0:
        logger.info("✅ Vehicle detection is working!")
    else:
        logger.info("⚠️ No vehicles detected - may need real vehicle images")

if __name__ == "__main__":
    quick_test()