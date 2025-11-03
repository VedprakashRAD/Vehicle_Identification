#!/usr/bin/env python3
"""
Mobile ANPR System for Workshop Vehicle Identification
Optimized for smartphone deployment with real-time processing
"""

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import sqlite3
from datetime import datetime
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileWorkshopANPR:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize mobile ANPR system with lightweight models"""
        # Use nano models for mobile performance
        self.vehicle_model = YOLO(model_path)
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU for mobile
        
        # VW Group brands
        self.vw_brands = ['Skoda', 'Audi', 'Porsche', 'Lamborghini', 'Bentley', 'Volkswagen']
        
        # Mobile-optimized settings
        self.frame_skip = 3  # Process every 3rd frame
        self.min_confidence = 0.5
        self.processing = False
        
        # Initialize database
        self.init_mobile_database()
        self.employee_plates = self.load_employee_plates()
        
    def init_mobile_database(self):
        """Initialize lightweight SQLite database for mobile"""
        conn = sqlite3.connect('mobile_workshop.db')
        cursor = conn.cursor()
        
        # Employee vehicles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                plate TEXT PRIMARY KEY,
                name TEXT,
                brand TEXT,
                department TEXT
            )
        ''')
        
        # Vehicle detections (simplified for mobile)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT,
                color TEXT,
                brand TEXT,
                timestamp DATETIME,
                is_employee BOOLEAN,
                location TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_employee_plates(self) -> Dict[str, Dict]:
        """Load employee data for fast lookup"""
        conn = sqlite3.connect('mobile_workshop.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT plate, name, brand, department FROM employees')
        results = cursor.fetchall()
        conn.close()
        
        return {row[0]: {'name': row[1], 'brand': row[2], 'dept': row[3]} for row in results}
    
    def add_employee(self, plate: str, name: str, brand: str, department: str):
        """Add employee vehicle"""
        conn = sqlite3.connect('mobile_workshop.db')
        cursor = conn.cursor()
        
        cursor.execute('INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?)', 
                      (plate, name, brand, department))
        conn.commit()
        conn.close()
        
        self.employee_plates[plate] = {'name': name, 'brand': brand, 'dept': department}
    
    def detect_color_mobile(self, roi: np.ndarray) -> str:
        """Lightweight color detection for mobile"""
        # Resize for faster processing
        small_roi = cv2.resize(roi, (64, 64))
        hsv = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
        
        # Simplified color ranges
        colors = {
            'White': [(0, 0, 200), (180, 30, 255)],
            'Black': [(0, 0, 0), (180, 255, 50)],
            'Red': [(0, 120, 70), (10, 255, 255)],
            'Blue': [(100, 150, 0), (140, 255, 255)]
        }
        
        max_pixels = 0
        detected_color = 'Unknown'
        
        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixels = cv2.countNonZero(mask)
            if pixels > max_pixels:
                max_pixels = pixels
                detected_color = color
                
        return detected_color
    
    def extract_plate_mobile(self, roi: np.ndarray) -> Optional[str]:
        """Mobile-optimized plate text extraction"""
        try:
            # Preprocess for better OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR with mobile settings
            results = self.ocr_reader.readtext(enhanced, detail=0, paragraph=False)
            
            if results:
                # Get longest text (likely the plate)
                plate_text = max(results, key=len).strip().upper()
                
                # Basic validation
                if len(plate_text) >= 6 and any(c.isdigit() for c in plate_text):
                    return plate_text
                    
        except Exception as e:
            logger.error(f"Mobile OCR error: {e}")
            
        return None
    
    def process_frame_mobile(self, frame: np.ndarray) -> List[Dict]:
        """Process frame with mobile optimizations"""
        detections = []
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Detect vehicles
        results = self.vehicle_model(frame, conf=self.min_confidence)
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Vehicle bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                vehicle_roi = frame[y1:y2, x1:x2]
                
                # Skip small detections
                if (x2-x1) * (y2-y1) < 5000:
                    continue
                
                # Try to find license plate in vehicle
                plate_text = self.find_plate_in_vehicle(vehicle_roi)
                
                if plate_text:
                    color = self.detect_color_mobile(vehicle_roi)
                    brand = self.classify_brand_mobile(vehicle_roi)
                    
                    detection = {
                        'plate': plate_text,
                        'color': color,
                        'brand': brand,
                        'bbox': [x1, y1, x2, y2],
                        'is_employee': plate_text in self.employee_plates,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    detections.append(detection)
                    
        return detections
    
    def find_plate_in_vehicle(self, vehicle_roi: np.ndarray) -> Optional[str]:
        """Find license plate within vehicle ROI"""
        # Look for rectangular regions that might be plates
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Check if contour could be a license plate
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Indian plates are roughly 2:1 to 4:1 ratio
            if 1.5 < aspect_ratio < 5 and w > 50 and h > 15:
                plate_roi = vehicle_roi[y:y+h, x:x+w]
                plate_text = self.extract_plate_mobile(plate_roi)
                
                if plate_text:
                    return plate_text
                    
        return None
    
    def classify_brand_mobile(self, roi: np.ndarray) -> str:
        """Simple brand classification for mobile"""
        # Placeholder - in production, use a lightweight CNN
        import random
        return random.choice(self.vw_brands)
    
    def log_detection_mobile(self, detection: Dict):
        """Log detection to mobile database"""
        conn = sqlite3.connect('mobile_workshop.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (plate, color, brand, timestamp, is_employee, location)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            detection['plate'],
            detection['color'], 
            detection['brand'],
            detection['timestamp'],
            detection['is_employee'],
            'Mobile Scan'
        ))
        
        conn.commit()
        conn.close()
    
    def start_camera_processing(self, camera_index: int = 0):
        """Start mobile camera processing"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
            
        # Set mobile-friendly resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for mobile
        
        self.processing = True
        frame_count = 0
        
        logger.info("Mobile ANPR started - Press 'q' to quit")
        
        while self.processing:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every nth frame
            if frame_count % self.frame_skip == 0:
                detections = self.process_frame_mobile(frame)
                
                # Draw detections on frame
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    
                    # Color code: Green for employees, Red for customers
                    color = (0, 255, 0) if detection['is_employee'] else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add text
                    label = f"{detection['plate']} ({'EMP' if detection['is_employee'] else 'CUST'})"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Log detection
                    self.log_detection_mobile(detection)
                    
                    # Print to console
                    status = "EMPLOYEE" if detection['is_employee'] else "CUSTOMER"
                    logger.info(f"{status}: {detection['plate']} ({detection['color']} {detection['brand']})")
            
            # Display frame
            cv2.imshow('Mobile Workshop ANPR', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.processing = False
    
    def get_mobile_stats(self) -> Dict:
        """Get statistics for mobile display"""
        conn = sqlite3.connect('mobile_workshop.db')
        cursor = conn.cursor()
        
        # Today's detections
        cursor.execute('''
            SELECT COUNT(*) FROM detections 
            WHERE DATE(timestamp) = DATE('now') AND is_employee = 0
        ''')
        customers_today = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(DISTINCT plate) FROM detections 
            WHERE DATE(timestamp) = DATE('now') AND is_employee = 1
        ''')
        employees_today = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'customers_today': customers_today,
            'employees_today': employees_today,
            'total_employees': len(self.employee_plates)
        }
    
    def export_mobile_data(self) -> str:
        """Export data as JSON for mobile sharing"""
        conn = sqlite3.connect('mobile_workshop.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT plate, color, brand, timestamp, is_employee 
            FROM detections 
            ORDER BY timestamp DESC LIMIT 100
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        data = []
        for row in results:
            data.append({
                'plate': row[0],
                'color': row[1],
                'brand': row[2],
                'timestamp': row[3],
                'type': 'Employee' if row[4] else 'Customer'
            })
            
        return json.dumps(data, indent=2)

# Mobile app simulation
if __name__ == "__main__":
    # Initialize mobile system
    mobile_anpr = MobileWorkshopANPR()
    
    # Add sample employees
    employees = [
        ('KA01AB1234', 'Rajesh Kumar', 'Skoda', 'Service'),
        ('KA02CD5678', 'Priya Sharma', 'Audi', 'Sales'),
        ('MH12GH3456', 'Amit Patel', 'Porsche', 'Management')
    ]
    
    for plate, name, brand, dept in employees:
        mobile_anpr.add_employee(plate, name, brand, dept)
    
    print("Mobile Workshop ANPR System Ready!")
    print(f"Loaded {len(employees)} employee vehicles")
    print("Starting camera... Press 'q' to quit")
    
    # Start processing
    mobile_anpr.start_camera_processing(camera_index=0)