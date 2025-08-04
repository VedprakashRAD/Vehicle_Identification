import cv2
import numpy as np
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available, using demo mode")

try:
    from license_plate.tiny_detector import TinyEnhancedDetector
    PLATE_DETECTION_AVAILABLE = True
except ImportError as e:
    PLATE_DETECTION_AVAILABLE = False
    logging.warning(f"Tiny detector not available: {e}")
    try:
        from license_plate.intelligent_detector import IntelligentLicensePlateDetector
        PLATE_DETECTION_AVAILABLE = True
    except ImportError:
        logging.warning("No plate detection available")

from config.settings import model_config, ui_config, llm_config

logger = logging.getLogger(__name__)

class WorkingVehicleTracker:
    """Enhanced vehicle tracking with improved modularity and performance"""
    
    def __init__(self, confidence_threshold: float = None):
        self.confidence_threshold = confidence_threshold or model_config.CONFIDENCE_THRESHOLD
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.total_count = 0
        self.active_tracks = 0
        self.entry_exit_log = deque(maxlen=100)  # Use deque for better performance
        self.latest_stats = {}
        self.model: Optional[YOLO] = None
        self.plate_detector = None
        self.license_plate_detections = deque(maxlen=50)  # Use deque for better performance
        self.frame_count = 0
        
        # Vehicle tracking with recent plates cache
        self.tracked_vehicles = {}
        self.vehicle_plates_cache = {}  # Cache recent plates per vehicle
        self.next_vehicle_id = 1
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize YOLO model and plate detector with proper error handling"""
        try:
            self._load_model()
            self._load_plate_detector()
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _load_model(self):
        """Load YOLO model with error handling"""
        if not YOLO_AVAILABLE:
            logger.info("YOLO not available, using demo mode")
            return
            
        try:
            logger.info("Loading YOLO model")
            self.model = YOLO(model_config.YOLO_MODEL_PATH)
            logger.info("YOLO model loaded successfully")
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_config.YOLO_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def _load_plate_detector(self):
        """Load TinyLLaVA enhanced detector"""
        if not PLATE_DETECTION_AVAILABLE:
            logger.info("License plate detection not available")
            return
            
        try:
            logger.info("Loading TinyLLaVA detector")
            
            try:
                self.plate_detector = TinyEnhancedDetector()
                logger.info("TinyLLaVA detector loaded (150MB)")
            except (ImportError, NameError):
                from license_plate.intelligent_detector import IntelligentLicensePlateDetector
                self.plate_detector = IntelligentLicensePlateDetector()
                logger.info("Using traditional detector")
                
        except Exception as e:
            logger.error(f"Error loading detector: {e}")
            self.plate_detector = None
        
    def process_frame_for_web(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame with YOLO detection and license plate recognition"""
        processed_frame = frame.copy()
        current_detections = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        
        self.frame_count += 1
        
        if self.model is not None:
            try:
                detections = self._run_vehicle_detection(frame)
                current_detections = self._process_detections(detections, processed_frame, frame)
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
                return self._demo_detection(processed_frame)
        else:
            return self._demo_detection(processed_frame)
        
        # Update statistics
        self._update_statistics(current_detections)
        
        # Add info overlay
        self._add_info_overlay(processed_frame, current_detections)
        
        # Prepare stats response
        stats = self._prepare_stats()
        self.latest_stats = stats
        
        return processed_frame, stats
    
    def _run_vehicle_detection(self, frame: np.ndarray) -> List:
        """Run YOLO inference on frame"""
        results = self.model(frame, conf=self.confidence_threshold)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in model_config.VEHICLE_CLASSES and confidence >= self.confidence_threshold:
                        vehicle_type = model_config.VEHICLE_CLASSES[class_id]
                        bbox = tuple(map(int, box.xyxy[0]))
                        
                        detections.append({
                            'vehicle_type': vehicle_type,
                            'bbox': bbox,
                            'confidence': confidence
                        })
        
        return detections
    
    def _process_detections(self, detections: List, processed_frame: np.ndarray, original_frame: np.ndarray) -> Dict:
        """Process vehicle detections and handle license plates"""
        current_detections = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        
        for detection in detections:
            vehicle_type = detection['vehicle_type']
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Track vehicle
            vehicle_id = self._track_vehicle(bbox, vehicle_type)
            
            # Draw vehicle detection
            self._draw_vehicle_detection(processed_frame, bbox, vehicle_type, vehicle_id, confidence)
            
            # Process license plate
            self._process_license_plate(original_frame, bbox, vehicle_id, vehicle_type, processed_frame)
            
            # Count detection
            current_detections[vehicle_type] += 1
        
        return current_detections
    
    def _draw_vehicle_detection(self, frame: np.ndarray, bbox: Tuple, vehicle_type: str, vehicle_id: str, confidence: float):
        """Draw vehicle detection on frame"""
        x1, y1, x2, y2 = bbox
        color = ui_config.COLORS.get(vehicle_type, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Get recent plate for this vehicle
        recent_plate = self.vehicle_plates_cache.get(vehicle_id, "")
        
        # Create label
        if recent_plate:
            label = f'{vehicle_type.upper()} {vehicle_id} | {recent_plate}'
        else:
            label = f'{vehicle_type.upper()} {vehicle_id} ({confidence:.2f})'
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _process_license_plate(self, frame: np.ndarray, bbox: Tuple, vehicle_id: str, vehicle_type: str, processed_frame: np.ndarray):
        """Process license plate detection for a vehicle"""
        if self.plate_detector is None:
            return
            
        try:
            plate_results = self.plate_detector.process_vehicle_for_plates(frame, bbox, vehicle_id)
            
            if plate_results:
                for plate_result in plate_results:
                    plate_text = plate_result.get('plate_text', '')
                    if plate_text:
                        # Update cache
                        self.vehicle_plates_cache[vehicle_id] = plate_text
                        
                        # Add to detections if unique
                        if not any(p.get('plate_text') == plate_text for p in list(self.license_plate_detections)[-5:]):
                            plate_result['vehicle_type'] = vehicle_type
                            self.license_plate_detections.append(plate_result)
                            logger.info(f"License plate detected: {plate_text} for {vehicle_type}")
                
                # Draw license plate detections
                self.plate_detector.draw_plate_detections(processed_frame, plate_results)
                
        except Exception as e:
            logger.error(f"Error in license plate detection: {e}")
    
    def _demo_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Fallback demo detection when YOLO is not available"""
        processed_frame = frame.copy()
        
        # Add demo detection boxes
        cv2.rectangle(processed_frame, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.putText(processed_frame, 'CAR (DEMO)', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update demo stats occasionally
        if self.frame_count % 30 == 0:
            self.vehicle_counts['car'] += 1
            self.total_count += 1
        
        self.active_tracks = 1
        
        stats = self._prepare_stats()
        return processed_frame, stats
    
    def _update_statistics(self, current_detections: Dict):
        """Update vehicle counting statistics"""
        self.active_tracks = sum(current_detections.values())
        
        # Update total counts (cumulative)
        for vehicle_type, count in current_detections.items():
            if count > 0:
                self.vehicle_counts[vehicle_type] += count
        
        self.total_count = sum(self.vehicle_counts.values())
    
    def _prepare_stats(self) -> Dict:
        """Prepare statistics dictionary"""
        return {
            'total_count': self.total_count,
            'vehicle_counts': self.vehicle_counts.copy(),
            'active_tracks': self.active_tracks,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_exit_log': list(self.entry_exit_log),
            'license_plates': self.get_recent_license_plates()
        }
    
    def _track_vehicle(self, bbox: Tuple, vehicle_type: str) -> str:
        """Track vehicle and assign unique ID with improved efficiency"""
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
    
    def _add_info_overlay(self, frame: np.ndarray, current_detections: Dict):
        """Add information overlay to frame with constants for positioning"""
        # Constants for overlay positioning
        OVERLAY_X, OVERLAY_Y = 10, 10
        OVERLAY_WIDTH, OVERLAY_HEIGHT = 350, 140
        TEXT_START_X, TEXT_START_Y = 20, 30
        LINE_HEIGHT = 25
        SMALL_LINE_HEIGHT = 20
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (OVERLAY_X, OVERLAY_Y), (OVERLAY_WIDTH, OVERLAY_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text with proper positioning
        y_pos = TEXT_START_Y
        cv2.putText(frame, f'Total: {self.total_count}', (TEXT_START_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += LINE_HEIGHT
        cv2.putText(frame, f'Active: {self.active_tracks}', (TEXT_START_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += LINE_HEIGHT
        
        # Vehicle counts with colors
        cv2.putText(frame, f'Cars: {self.vehicle_counts["car"]}', (TEXT_START_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_config.COLORS['car'], 1)
        cv2.putText(frame, f'Motorcycles: {self.vehicle_counts["motorcycle"]}', (TEXT_START_X + 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_config.COLORS['motorcycle'], 1)
        y_pos += SMALL_LINE_HEIGHT
        cv2.putText(frame, f'Buses: {self.vehicle_counts["bus"]}', (TEXT_START_X, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_config.COLORS['bus'], 1)
        cv2.putText(frame, f'Trucks: {self.vehicle_counts["truck"]}', (TEXT_START_X + 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_config.COLORS['truck'], 1)
        
    def get_recent_license_plates(self, limit: int = 10) -> List[Dict]:
        """Get recent license plate detections"""
        if self.plate_detector:
            return self.plate_detector.get_recent_detections(limit)
        return []
    
    def reset_counts(self):
        """Reset all counters and tracking data"""
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.total_count = 0
        self.active_tracks = 0
        self.entry_exit_log.clear()
        self.license_plate_detections.clear()
        self.tracked_vehicles.clear()
        self.vehicle_plates_cache.clear()
        self.next_vehicle_id = 1
        self.frame_count = 0
        logger.info("Vehicle tracking counters reset")
        
    def get_vehicle_details(self) -> List[Dict]:
        """Get vehicle details including license plates"""
        details = []
        recent_plates = self.get_recent_license_plates(20)
        
        logger.info(f"Processing {len(recent_plates)} recent plates for vehicle details")
        
        # Create vehicle details with license plate info
        for i, plate in enumerate(recent_plates):
            vehicle_type = plate.get('vehicle_type', 'Car')
            confidence = plate.get('confidence', 0.0)
            
            try:
                timestamp = datetime.fromisoformat(plate['timestamp'])
                entry_time = timestamp.strftime('%H:%M:%S')
            except (KeyError, ValueError):
                entry_time = datetime.now().strftime('%H:%M:%S')
            
            details.append({
                'vehicle_id': plate.get('vehicle_id', f'V{i+1}'),
                'registration_number': plate.get('plate_text', 'Unknown'),
                'vehicle_type': vehicle_type.title(),
                'status': 'Entry',
                'entry_time': entry_time,
                'exit_time': None,
                'confidence': confidence
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
        
        logger.info(f"Created {len(details)} vehicle detail entries")
        return details