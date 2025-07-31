import cv2
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO not available, using demo mode")

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import license plate detector
try:
    from license_plate.intelligent_detector import IntelligentLicensePlateDetector
    PLATE_DETECTION_AVAILABLE = True
except ImportError as e:
    PLATE_DETECTION_AVAILABLE = False
    print(f"⚠️ License plate detection not available: {e}")

class WorkingVehicleTracker:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.total_count = 0
        self.active_tracks = 0
        self.entry_exit_log = []
        self.latest_stats = {}
        self.model = None
        self.plate_detector = None
        self.license_plate_detections = []
        self.load_model()
        self.load_plate_detector()
        
        # Vehicle class mapping (COCO dataset classes for YOLOv8)
        self.vehicle_classes = {
            2: 'car',        # car
            3: 'motorcycle', # motorcycle  
            5: 'bus',        # bus
            7: 'truck'       # truck
        }
        
        # Vehicle tracking
        self.tracked_vehicles = {}  # Store vehicle positions and IDs
        self.next_vehicle_id = 1
        self.frame_count = 0  # Add frame counter for debugging
        # Remove processed_vehicles tracking to allow continuous plate detection
        
    def load_model(self):
        """Load YOLO model"""
        if not YOLO_AVAILABLE:
            print("🔄 YOLO not available, using demo mode")
            self.model = None
            return
            
        try:
            print("📦 Loading YOLOv8n model")
            self.model = YOLO('yolov8n.pt')  # YOLOv8 nano model
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Falling back to demo mode")
            self.model = None
    
    def load_plate_detector(self):
        """Load license plate detector"""
        if PLATE_DETECTION_AVAILABLE:
            try:
                print("📦 Loading license plate detector...")
                self.plate_detector = IntelligentLicensePlateDetector()
                print("✅ License plate detector loaded successfully")
            except Exception as e:
                print(f"❌ Error loading plate detector: {e}")
                self.plate_detector = None
        else:
            print("⚠️ License plate detection not available")
            self.plate_detector = None
        
    def process_frame_for_web(self, frame):
        """Process frame with YOLOv5 detection"""
        processed_frame = frame.copy()
        current_detections = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        
        if self.model is not None:
            try:
                self.frame_count += 1
                # Run inference
                results = self.model(frame, conf=self.confidence_threshold)
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get class ID and confidence
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Check if it's a vehicle class
                            if class_id in self.vehicle_classes and confidence >= self.confidence_threshold:
                                vehicle_type = self.vehicle_classes[class_id]
                                
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Track vehicle and get unique ID
                                vehicle_id = self.track_vehicle((x1, y1, x2, y2), vehicle_type)
                                
                                # Draw bounding box
                                color = self.get_color_for_vehicle(vehicle_type)
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Get recent plate for this vehicle
                                recent_plate = ""
                                if self.plate_detector:
                                    recent_detections = self.plate_detector.get_recent_detections(10)
                                    for detection in reversed(recent_detections):
                                        if detection.get('vehicle_id') == vehicle_id:
                                            recent_plate = detection.get('plate_text', '')
                                            break
                                
                                # Draw label with vehicle ID and plate
                                if recent_plate:
                                    label = f'{vehicle_type.upper()} {vehicle_id} | {recent_plate}'
                                else:
                                    label = f'{vehicle_type.upper()} {vehicle_id} ({confidence:.2f})'
                                
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                                cv2.putText(processed_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                # Process license plate for every detection (not just once per vehicle)
                                if self.plate_detector is not None:
                                    try:
                                        plate_results = self.plate_detector.process_vehicle_for_plates(
                                            frame, (x1, y1, x2, y2), vehicle_id
                                        )
                                        
                                        if plate_results:
                                            print(f"🎯 License plate detected for vehicle {vehicle_id}: {len(plate_results)} plates")
                                            # Only add unique plates to avoid duplicates
                                            for plate_result in plate_results:
                                                plate_text = plate_result.get('plate_text', '')
                                                if plate_text and not any(p.get('plate_text') == plate_text for p in self.license_plate_detections[-5:]):
                                                    self.license_plate_detections.append(plate_result)
                                                    print(f"📝 Added license plate: {plate_text}")
                                            
                                            # Draw license plate detections
                                            processed_frame = self.plate_detector.draw_plate_detections(
                                                processed_frame, plate_results
                                            )
                                        else:
                                            # Debug: Try to understand why no plates detected
                                            if self.frame_count % 30 == 0:  # Log every 30 frames to avoid spam
                                                print(f"🔍 No license plates detected for vehicle {vehicle_id} at bbox {(x1, y1, x2, y2)}")
                                    except Exception as e:
                                        print(f"❌ Error in license plate detection: {e}")
                                
                                # Count detection
                                current_detections[vehicle_type] += 1
                
                # Update active tracks
                self.active_tracks = sum(current_detections.values())
                
                # Update total counts (cumulative)
                for vehicle_type, count in current_detections.items():
                    if count > 0:
                        self.vehicle_counts[vehicle_type] += count
                
                self.total_count = sum(self.vehicle_counts.values())
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
                # Fall back to demo mode
                return self.demo_detection(processed_frame)
        else:
            # Demo mode
            return self.demo_detection(processed_frame)
        
        # Add info overlay
        self.add_info_overlay(processed_frame, current_detections)
        
        stats = {
            'total_count': self.total_count,
            'vehicle_counts': self.vehicle_counts.copy(),
            'active_tracks': self.active_tracks,
            'timestamp': datetime.now().isoformat(),
            'entry_exit_log': self.entry_exit_log,
            'license_plates': self.get_recent_license_plates()
        }
        
        self.latest_stats = stats
        return processed_frame, stats
    
    def demo_detection(self, frame):
        """Fallback demo detection"""
        processed_frame = frame.copy()
        
        # Add demo detection boxes
        cv2.rectangle(processed_frame, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.putText(processed_frame, 'CAR (DEMO)', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update demo stats occasionally
        if self.total_count % 30 == 0:  # Update every 30 frames
            self.vehicle_counts['car'] += 1
            self.total_count += 1
        
        self.active_tracks = 1
        
        stats = {
            'total_count': self.total_count,
            'vehicle_counts': self.vehicle_counts.copy(),
            'active_tracks': self.active_tracks,
            'timestamp': datetime.now().isoformat(),
            'entry_exit_log': self.entry_exit_log
        }
        
        return processed_frame, stats
    
    def get_color_for_vehicle(self, vehicle_type):
        """Get color for vehicle type"""
        colors = {
            'car': (0, 255, 0),      # Green
            'motorcycle': (255, 0, 0), # Blue
            'bus': (0, 255, 255),    # Yellow
            'truck': (255, 0, 255)   # Magenta
        }
        return colors.get(vehicle_type, (255, 255, 255))
    
    def track_vehicle(self, bbox, vehicle_type):
        """Track vehicle and assign unique ID"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Find existing vehicle within threshold
        threshold = 50
        for vid, data in self.tracked_vehicles.items():
            old_x, old_y = data['center']
            if abs(center_x - old_x) < threshold and abs(center_y - old_y) < threshold:
                # Update position
                self.tracked_vehicles[vid]['center'] = (center_x, center_y)
                self.tracked_vehicles[vid]['bbox'] = bbox
                return vid
        
        # New vehicle
        vehicle_id = f"V{self.next_vehicle_id}"
        self.tracked_vehicles[vehicle_id] = {
            'center': (center_x, center_y),
            'bbox': bbox,
            'type': vehicle_type
        }
        self.next_vehicle_id += 1
        return vehicle_id
    
    def add_info_overlay(self, frame, current_detections):
        """Add information overlay to frame"""
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        y_offset = 30
        cv2.putText(frame, f'Total: {self.total_count}', (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(frame, f'Active: {self.active_tracks}', (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f'Cars: {self.vehicle_counts["car"]}', (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f'Motorcycles: {self.vehicle_counts["motorcycle"]}', (120, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y_offset += 20
        cv2.putText(frame, f'Buses: {self.vehicle_counts["bus"]}', (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f'Trucks: {self.vehicle_counts["truck"]}', (120, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
    def get_recent_license_plates(self, limit=10):
        """Get recent license plate detections"""
        if self.plate_detector:
            return self.plate_detector.get_recent_detections(limit)
        return []
    
    def reset_counts(self):
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.total_count = 0
        self.active_tracks = 0
        self.entry_exit_log = []
        self.license_plate_detections = []
        self.tracked_vehicles = {}
        self.next_vehicle_id = 1
        self.frame_count = 0
        # Removed processed_vehicles to allow continuous detection
        
    def get_vehicle_details(self):
        """Get vehicle details including license plates"""
        details = []
        recent_plates = self.get_recent_license_plates(20)
        
        # Create vehicle details with license plate info
        for i, plate in enumerate(recent_plates):
            details.append({
                'vehicle_id': plate.get('vehicle_id', f'V{i+1}'),
                'registration_number': plate.get('plate_text', 'Unknown'),
                'vehicle_type': 'Car',  # Default, could be enhanced
                'status': 'Entry',
                'entry_time': datetime.fromisoformat(plate['timestamp']).strftime('%H:%M:%S'),
                'exit_time': None,
                'confidence': plate.get('confidence', 0.0)
            })
        
        # Add demo data if no plates detected
        if not details:
            details.append({
                'vehicle_id': 'V001',
                'registration_number': 'ABC-123',
                'vehicle_type': 'Car',
                'status': 'Entry',
                'entry_time': datetime.now().strftime('%H:%M:%S'),
                'exit_time': None,
                'confidence': 0.95
            })
        
        return details