"""
Vehicle Pairing Module
Handles pairing of front and rear license plates for entry/exit events
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class VehiclePairingManager:
    """Manages pairing of front and rear license plates for vehicle entry/exit tracking"""
    
    def __init__(self, db_path='vehicle_data.db'):
        self.db_path = db_path
        self.pairing_window = timedelta(seconds=30)  # Time window for pairing plates
        self._init_pairing_table()
    
    def _init_pairing_table(self):
        """Initialize the temporary pairing table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table to store temporary plate detections for pairing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plate_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                camera_id INTEGER,  -- 1 for entry camera, 2 for exit camera
                vehicle_color TEXT,
                vehicle_brand TEXT,
                detection_time DATETIME,
                is_employee BOOLEAN,
                image_path TEXT
            )
        ''')
        
        # Table to store paired vehicle events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                front_plate TEXT,
                rear_plate TEXT,
                entry_time DATETIME,
                exit_time DATETIME,
                entry_camera_id INTEGER,
                exit_camera_id INTEGER,
                vehicle_color TEXT,
                vehicle_brand TEXT,
                is_employee BOOLEAN,
                status TEXT,  -- 'entry_pending', 'exit_pending', 'completed', 'anomaly'
                visit_count INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_plate_detection(self, plate_number: str, camera_id: int, vehicle_color: str = 'unknown', 
                           vehicle_brand: str = 'unknown', is_employee: bool = False, 
                           image_path: str = None) -> bool:
        """
        Add a plate detection for pairing
        Returns True if successfully added, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO plate_detections 
                (plate_number, camera_id, vehicle_color, vehicle_brand, detection_time, is_employee, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (plate_number, camera_id, vehicle_color, vehicle_brand, datetime.now(), is_employee, image_path))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added plate detection: {plate_number} from camera {camera_id}")
            
            # Try to pair this detection
            self._attempt_pairing()
            
            return True
        except Exception as e:
            logger.error(f"Error adding plate detection: {e}")
            return False
    
    def _attempt_pairing(self):
        """Attempt to pair plate detections into complete vehicle events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent unpaired detections
            cursor.execute('''
                SELECT id, plate_number, camera_id, vehicle_color, vehicle_brand, detection_time, is_employee
                FROM plate_detections
                WHERE detection_time > ?
                ORDER BY detection_time DESC
            ''', (datetime.now() - self.pairing_window,))
            
            detections = cursor.fetchall()
            
            # Group detections by time windows
            time_groups = self._group_detections_by_time(detections)
            
            # Attempt to pair detections in each time group
            for group in time_groups:
                self._pair_group_detections(group, cursor, conn)
            
            conn.close()
        except Exception as e:
            logger.error(f"Error in pairing attempt: {e}")
    
    def _group_detections_by_time(self, detections: List) -> List[List]:
        """Group detections by time windows for pairing"""
        if not detections:
            return []
        
        # Sort by time (convert string to datetime if needed)
        def parse_time(detection):
            time_val = detection[5]
            if isinstance(time_val, str):
                # Parse string datetime
                try:
                    return datetime.fromisoformat(time_val)
                except:
                    return datetime.strptime(time_val, '%Y-%m-%d %H:%M:%S.%f')
            return time_val
        
        detections.sort(key=lambda x: parse_time(x))
        
        groups = []
        current_group = [detections[0]]
        
        for detection in detections[1:]:
            time1 = parse_time(detection)
            time2 = parse_time(current_group[-1])
            time_diff = abs((time1 - time2).total_seconds())
            
            # If time difference is within window, add to current group
            if time_diff <= self.pairing_window.total_seconds():
                current_group.append(detection)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [detection]
        
        groups.append(current_group)
        return groups
    
    def _pair_group_detections(self, group: List, cursor, conn):
        """Pair detections within a time group using enhanced logic for entry/exit"""
        # Convert string times to datetime objects if needed
        def parse_detection_time(detection):
            time_val = detection[5]
            if isinstance(time_val, str):
                try:
                    return datetime.fromisoformat(time_val)
                except:
                    return datetime.strptime(time_val, '%Y-%m-%d %H:%M:%S.%f')
            return time_val

        # Parse times for all detections
        parsed_group = []
        for detection in group:
            parsed_det = list(detection)
            parsed_det[5] = parse_detection_time(detection)
            parsed_group.append(tuple(parsed_det))

        # Separate by camera
        camera1_detections = [d for d in parsed_group if d[2] == 1]  # Camera 1 (entry camera)
        camera2_detections = [d for d in parsed_group if d[2] == 2]  # Camera 2 (exit camera)

        # Try to pair detections from both cameras
        if camera1_detections and camera2_detections:
            # Create all possible pairs and sort by time difference
            pairs = []
            for cam1_det in camera1_detections:
                for cam2_det in camera2_detections:
                    time_diff = abs((cam1_det[5] - cam2_det[5]).total_seconds())
                    
                    # Only consider pairs within the pairing window
                    if time_diff <= self.pairing_window.total_seconds():
                        # Check if vehicle attributes match (color and brand)
                        color_match = cam1_det[3] == cam2_det[3] or cam1_det[3] == 'unknown' or cam2_det[3] == 'unknown'
                        brand_match = cam1_det[4] == cam2_det[4] or cam1_det[4] == 'unknown' or cam2_det[4] == 'unknown'
                        
                        # If attributes match or are unknown, consider this pair
                        if color_match and brand_match:
                            pairs.append((cam1_det, cam2_det, time_diff))
            
            # Sort pairs by time difference (closest in time first)
            pairs.sort(key=lambda x: x[2])
            
            # Process pairs in order of time proximity
            processed_detections = set()
            for cam1_det, cam2_det, time_diff in pairs:
                # Skip if either detection has already been processed
                if cam1_det[0] in processed_detections or cam2_det[0] in processed_detections:
                    continue
                
                # Create paired event
                self._create_paired_event(cam1_det, cam2_det, cursor, conn)
                processed_detections.add(cam1_det[0])
                processed_detections.add(cam2_det[0])
        elif camera1_detections or camera2_detections:
            # If we only have detections from one camera, check if they can complete existing events
            all_detections = camera1_detections + camera2_detections
            for detection in all_detections:
                self._try_complete_existing_event(detection, cursor, conn)
    
    def _try_complete_existing_event(self, detection, cursor, conn):
        """Try to complete an existing event with a single detection"""
        plate_number = detection[1]
        camera_id = detection[2]
        detection_time = detection[5]
        
        # Convert datetime to string for database storage
        detection_time_str = detection_time.isoformat() if hasattr(detection_time, 'isoformat') else str(detection_time)
        
        # If this is from Camera 1 (entry camera)
        if camera_id == 1:
            # Look for incomplete events that might be completed by this front plate detection
            cursor.execute('''
                SELECT id, front_plate, rear_plate FROM vehicle_events 
                WHERE front_plate = ? AND exit_time IS NULL
                ORDER BY entry_time DESC
                LIMIT 1
            ''', (plate_number,))
            
            result = cursor.fetchone()
            if result:
                # This might be an exit completion, but we need a rear plate detection for that
                # For now, we'll leave it as is since Camera 1 should detect front plates
                pass
        
        # If this is from Camera 2 (exit camera)
        elif camera_id == 2:
            # Look for incomplete events that might be completed by this rear plate detection
            cursor.execute('''
                SELECT id, front_plate, rear_plate FROM vehicle_events 
                WHERE rear_plate = ? AND exit_time IS NULL
                ORDER BY entry_time DESC
                LIMIT 1
            ''', (plate_number,))
            
            result = cursor.fetchone()
            if result:
                # Complete this event with the rear plate detection
                event_id, front_plate, rear_plate = result
                
                cursor.execute('''
                    UPDATE vehicle_events 
                    SET exit_time = ?, exit_camera_id = ?, status = 'completed'
                    WHERE id = ?
                ''', (detection_time_str, camera_id, event_id))
                
                logger.info(f"Vehicle EXIT completed: Front={front_plate}, Rear={rear_plate} at {detection_time_str}")
                
                # Remove the detection from temporary table
                cursor.execute('DELETE FROM plate_detections WHERE id = ?', (detection[0],))
                conn.commit()
    
    def _create_paired_event(self, cam1_det, cam2_det, cursor, conn):
        """Create a paired vehicle event with proper entry/exit logic"""
        # For entry event:
        # - Camera 1 (entry camera) detects front plate first
        # - Camera 2 (exit camera) detects rear plate second
        #
        # For exit event:
        # - Camera 2 (exit camera) detects rear plate first (vehicle leaving)
        # - Camera 1 (entry camera) detects front plate second
        
        cam1_time = cam1_det[5]  # Camera 1 time
        cam2_time = cam2_det[5]  # Camera 2 time
        
        # Convert datetime to string for database storage
        cam1_time_str = cam1_time.isoformat() if hasattr(cam1_time, 'isoformat') else str(cam1_time)
        cam2_time_str = cam2_time.isoformat() if hasattr(cam2_time, 'isoformat') else str(cam2_time)
        
        # Determine if this is an entry or exit based on plate numbers and timing
        plate1 = cam1_det[1]  # Plate from Camera 1
        plate2 = cam2_det[1]  # Plate from Camera 2
        
        # Check if this is an employee vehicle
        is_employee = cam1_det[6] or cam2_det[6]
        
        # For entry: Camera 1 detects front plate, Camera 2 detects rear plate
        # We'll assume Camera 1 detection is the front plate and Camera 2 is the rear plate
        front_plate = plate1
        rear_plate = plate2
        entry_time = cam1_time_str
        exit_time = cam2_time_str if cam2_time > cam1_time else None
        
        # Check if we already have an incomplete entry for this plate combination
        cursor.execute('''
            SELECT id FROM vehicle_events 
            WHERE front_plate = ? AND rear_plate = ? AND exit_time IS NULL
        ''', (front_plate, rear_plate))
        
        existing_entry = cursor.fetchone()
        
        if existing_entry and exit_time:
            # Complete the existing entry with exit information
            cursor.execute('''
                UPDATE vehicle_events 
                SET exit_time = ?, exit_camera_id = ?, status = 'completed'
                WHERE id = ?
            ''', (exit_time, cam2_det[2], existing_entry[0]))
            
            logger.info(f"Vehicle journey COMPLETED: Front={front_plate}, Rear={rear_plate}")
        elif not existing_entry:
            # Create new entry event (incomplete journey)
            cursor.execute('''
                INSERT INTO vehicle_events 
                (front_plate, rear_plate, entry_time, entry_camera_id, vehicle_color, vehicle_brand, is_employee, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (front_plate, rear_plate, entry_time, cam1_det[2], 
                  cam1_det[3], cam1_det[4], is_employee, 'entry_pending'))
            
            logger.info(f"New vehicle ENTRY recorded: Front={front_plate}, Rear={rear_plate} at {entry_time}")
        
        # Remove paired detections from temporary table
        cursor.execute('DELETE FROM plate_detections WHERE id IN (?, ?)', (cam1_det[0], cam2_det[0]))
        conn.commit()
    
    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        """Get recent vehicle events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, front_plate, rear_plate, entry_time, exit_time, 
                       entry_camera_id, exit_camera_id, vehicle_color, vehicle_brand, 
                       is_employee, status, visit_count
                FROM vehicle_events
                ORDER BY COALESCE(entry_time, exit_time) DESC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            events = []
            for row in results:
                events.append({
                    'id': row[0],
                    'front_plate': row[1],
                    'rear_plate': row[2],
                    'entry_time': row[3],
                    'exit_time': row[4],
                    'entry_camera_id': row[5],
                    'exit_camera_id': row[6],
                    'vehicle_color': row[7],
                    'vehicle_brand': row[8],
                    'is_employee': row[9],
                    'status': row[10],
                    'visit_count': row[11]
                })
            
            return events
        except Exception as e:
            logger.error(f"Error fetching recent events: {e}")
            return []
    
    def check_for_anomalies(self) -> List[Dict]:
        """Check for inconsistent or anomalous events"""
        anomalies = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for events with missing plates
            cursor.execute('''
                SELECT id, front_plate, rear_plate, entry_time, exit_time
                FROM vehicle_events
                WHERE front_plate IS NULL OR rear_plate IS NULL
            ''')
            
            missing_plates = cursor.fetchall()
            for row in missing_plates:
                anomalies.append({
                    'type': 'missing_plate',
                    'event_id': row[0],
                    'front_plate': row[1],
                    'rear_plate': row[2],
                    'entry_time': row[3],
                    'exit_time': row[4],
                    'message': 'Event missing front or rear plate number'
                })
            
            # Check for events with mismatched timing
            cursor.execute('''
                SELECT id, front_plate, rear_plate, entry_time, exit_time
                FROM vehicle_events
                WHERE exit_time IS NOT NULL AND entry_time IS NOT NULL AND entry_time > exit_time
            ''')
            
            time_mismatches = cursor.fetchall()
            for row in time_mismatches:
                anomalies.append({
                    'type': 'time_mismatch',
                    'event_id': row[0],
                    'front_plate': row[1],
                    'rear_plate': row[2],
                    'entry_time': row[3],
                    'exit_time': row[4],
                    'message': 'Exit time is before entry time'
                })
            
            # Check for long-staying vehicles (over 24 hours)
            cursor.execute('''
                SELECT id, front_plate, rear_plate, entry_time, exit_time
                FROM vehicle_events
                WHERE exit_time IS NOT NULL AND entry_time IS NOT NULL
                AND (strftime('%s', exit_time) - strftime('%s', entry_time)) > 86400
            ''')
            
            long_stays = cursor.fetchall()
            for row in long_stays:
                anomalies.append({
                    'type': 'long_stay',
                    'event_id': row[0],
                    'front_plate': row[1],
                    'rear_plate': row[2],
                    'entry_time': row[3],
                    'exit_time': row[4],
                    'message': 'Vehicle stayed for more than 24 hours'
                })
            
            # Check for orphaned exit events (exit without entry)
            cursor.execute('''
                SELECT id, front_plate, rear_plate, entry_time, exit_time
                FROM vehicle_events
                WHERE entry_time IS NULL AND exit_time IS NOT NULL
            ''')
            
            orphaned_exits = cursor.fetchall()
            for row in orphaned_exits:
                anomalies.append({
                    'type': 'orphaned_exit',
                    'event_id': row[0],
                    'front_plate': row[1],
                    'rear_plate': row[2],
                    'entry_time': row[3],
                    'exit_time': row[4],
                    'message': 'Vehicle exit detected without corresponding entry'
                })
            
            conn.close()
        except Exception as e:
            logger.error(f"Error checking for anomalies: {e}")
        
        return anomalies
    
    def cleanup_old_detections(self):
        """Clean up old unpaired detections"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete detections older than 1 hour
            cursor.execute('''
                DELETE FROM plate_detections
                WHERE detection_time < ?
            ''', (datetime.now() - timedelta(hours=1),))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error cleaning up old detections: {e}")