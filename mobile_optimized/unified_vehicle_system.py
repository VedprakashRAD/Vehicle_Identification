#!/usr/bin/env python3
"""
Unified Vehicle Identification System
Works on Mobile, Web Dashboard, and Standalone App
"""

import argparse
import sys
import os
import cv2
import numpy as np
from datetime import datetime
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedVehicleSystem:
    def __init__(self):
        self.mode = None
        self.is_running = False
        
    def run_web_dashboard(self, host='0.0.0.0', port=9090):
        """Run the web dashboard"""
        try:
            logger.info("Starting Web Dashboard...")
            # Change to the main project directory to ensure proper imports
            project_root = Path(__file__).parent.parent
            os.chdir(project_root)
            
            # Add project root to Python path
            sys.path.insert(0, str(project_root))
            
            # Import and run the web dashboard
            from web.dashboard import VehicleDashboard
            from config.settings import app_config
            
            # Update config with provided parameters
            app_config.HOST = host
            app_config.PORT = port
            
            dashboard = VehicleDashboard(
                host=host,
                port=port,
                debug=True
            )
            
            logger.info(f"Web Dashboard starting on {host}:{port}")
            dashboard.run()
            
        except Exception as e:
            logger.error(f"Error starting web dashboard: {e}")
            raise
            
    def run_mobile_anpr(self, camera_index=0):
        """Run mobile ANPR system"""
        try:
            logger.info("Starting Mobile ANPR System...")
            
            # Change to mobile_optimized directory
            mobile_dir = Path(__file__).parent
            os.chdir(mobile_dir)
            
            # Add to Python path
            sys.path.insert(0, str(mobile_dir))
            
            # Import and run mobile ANPR
            from mobile_anpr import MobileWorkshopANPR
            
            # Initialize mobile system
            mobile_anpr = MobileWorkshopANPR()
            
            # Add sample employees if database is empty
            employees = [
                ('KA01AB1234', 'Rajesh Kumar', 'Skoda', 'Service'),
                ('KA02CD5678', 'Priya Sharma', 'Audi', 'Sales'),
                ('MH12GH3456', 'Amit Patel', 'Porsche', 'Management')
            ]
            
            for plate, name, brand, dept in employees:
                mobile_anpr.add_employee(plate, name, brand, dept)
            
            logger.info(f"Loaded {len(employees)} employee vehicles")
            
            # Start processing
            success = mobile_anpr.start_camera_processing(camera_index=camera_index)
            
            if success:
                logger.info("Mobile ANPR system completed successfully")
            else:
                logger.error("Mobile ANPR system encountered an error")
                
        except Exception as e:
            logger.error(f"Error starting mobile ANPR: {e}")
            raise
            
    def run_standalone_app(self, source=0):
        """Run standalone application"""
        try:
            logger.info("Starting Standalone Application...")
            
            # Change to main project directory
            project_root = Path(__file__).parent.parent
            os.chdir(project_root)
            
            # Add to Python path
            sys.path.insert(0, str(project_root))
            
            # Import required modules
            from core.working_tracker import WorkingVehicleTracker
            
            # Initialize camera
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error("Cannot open camera")
                return
                
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize vehicle tracker
            tracker = WorkingVehicleTracker()
            
            logger.info("Standalone app started. Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame, stats = tracker.process_frame_for_web(frame)
                
                # Display frame
                cv2.imshow('Vehicle Identification System', processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Error in standalone app: {e}")
            raise
            
    def run(self, mode='web', **kwargs):
        """Run the system in specified mode"""
        self.mode = mode
        self.is_running = True
        
        if mode == 'web':
            self.run_web_dashboard(**kwargs)
        elif mode == 'mobile':
            self.run_mobile_anpr(**kwargs)
        elif mode == 'standalone':
            self.run_standalone_app(**kwargs)
        else:
            logger.error(f"Unknown mode: {mode}")
            raise ValueError(f"Unknown mode: {mode}")

def print_system_info():
    """Print system information"""
    print("=" * 50)
    print("🚗 Unified Vehicle Identification System")
    print("=" * 50)
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__ if 'cv2' in globals() else 'Not loaded'}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Unified Vehicle Identification System')
    parser.add_argument('--mode', choices=['web', 'mobile', 'standalone'], default='web',
                       help='Run mode: web dashboard, mobile app, or standalone')
    parser.add_argument('--host', default='0.0.0.0', help='Host for web dashboard')
    parser.add_argument('--port', type=int, default=9090, help='Port for web dashboard')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for mobile/standalone')
    parser.add_argument('--source', type=int, default=0, help='Video source for standalone app')
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    print(f"Mode: {args.mode}")
    
    # Initialize and run system
    system = UnifiedVehicleSystem()
    
    try:
        if args.mode == 'web':
            print(f"🌐 Web Dashboard will be available at: http://{args.host}:{args.port}")
            print("💡 Press Ctrl+C to stop")
            system.run(mode='web', host=args.host, port=args.port)
        elif args.mode == 'mobile':
            print("📱 Mobile ANPR System Starting...")
            print("📷 Camera index:", args.camera)
            print("💡 Press 'q' to quit")
            system.run(mode='mobile', camera_index=args.camera)
        elif args.mode == 'standalone':
            print("🖥️  Standalone Application Starting...")
            print("📷 Source:", args.source)
            print("💡 Press 'q' to quit")
            system.run(mode='standalone', source=args.source)
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
        logger.info("System stopped by user")
    except Exception as e:
        print(f"❌ Error running system: {e}")
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()