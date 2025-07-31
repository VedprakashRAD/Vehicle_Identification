"""
Advanced License Plate Detection and Recognition Module
"""

import cv2
import numpy as np
import re
import os
from datetime import datetime
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class AdvancedLicensePlateDetector:
    def __init__(self):
        self.plate_model = None
        self.ocr_reader = None
        self.detection_history = []
        self.vehicle_readings = {}  # Store multiple readings per vehicle
        self.load_models()
        
    def load_models(self):
        """Load detection and OCR models"""
        if YOLO_AVAILABLE:
            try:
                if os.path.exists('yolov8_license_plate2 (1).pt'):
                    self.plate_model = YOLO('yolov8_license_plate2 (1).pt')
                else:
                    self.plate_model = YOLO('yolov8n.pt')
                print("✅ YOLO model loaded")
            except:
                self.plate_model = None
        
        if OCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR loaded")
            except:
                self.ocr_reader = None
    
    def detect_plates(self, image, vehicle_bbox):
        """Enhanced plate detection"""
        plates = []
        
        # Focus on vehicle region
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_region = image[y1:y2, x1:x2]
        
        # Multiple detection methods
        methods = [
            self.detect_with_yolo,
            self.detect_with_contours,
            self.detect_with_edges,
            self.detect_with_morphology
        ]
        
        for method in methods:
            try:
                method_plates = method(vehicle_region, (x1, y1))
                plates.extend(method_plates)
            except:
                continue
        
        # Remove duplicates
        unique_plates = []
        for plate in plates:
            is_duplicate = False
            for existing in unique_plates:
                if self.boxes_overlap(plate['bbox'], existing['bbox'], 0.5):
                    if plate['confidence'] > existing['confidence']:
                        unique_plates.remove(existing)
                    else:
                        is_duplicate = True
                    break
            if not is_duplicate:
                unique_plates.append(plate)
        
        return unique_plates[:3]  # Top 3 candidates
    
    def detect_with_yolo(self, image, offset):
        """YOLO-based detection"""
        plates = []
        if self.plate_model:
            results = self.plate_model(image, conf=0.2)
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        plates.append({
                            'bbox': (x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]),
                            'confidence': conf,
                            'image': image[y1:y2, x1:x2]
                        })
        return plates
    
    def detect_with_contours(self, image, offset):
        """Contour-based detection"""
        plates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 8000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 2.0 < aspect_ratio < 6.0:
                    plates.append({
                        'bbox': (x + offset[0], y + offset[1], x + w + offset[0], y + h + offset[1]),
                        'confidence': 0.6,
                        'image': image[y:y+h, x:x+w]
                    })
        return plates
    
    def detect_with_edges(self, image, offset):
        """Edge-based detection"""
        plates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
        
        # Find rectangular regions
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 400 < area < 6000:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                x, y, w, h = cv2.boundingRect(box)
                if w > h and w/h > 1.5:
                    plates.append({
                        'bbox': (x + offset[0], y + offset[1], x + w + offset[0], y + h + offset[1]),
                        'confidence': 0.5,
                        'image': image[y:y+h, x:x+w]
                    })
        return plates
    
    def detect_with_morphology(self, image, offset):
        """Morphological operations detection"""
        plates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 600 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 2.5 < aspect_ratio < 5.0:
                    plates.append({
                        'bbox': (x + offset[0], y + offset[1], x + w + offset[0], y + h + offset[1]),
                        'confidence': 0.7,
                        'image': image[y:y+h, x:x+w]
                    })
        return plates
    
    def recognize_text_advanced(self, plate_image):
        """Advanced OCR with multiple engines and preprocessing"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Generate multiple image variants
        variants = self.create_image_variants(plate_image)
        
        results = []
        
        # EasyOCR
        if self.ocr_reader:
            for variant in variants:
                try:
                    ocr_results = self.ocr_reader.readtext(variant, detail=1)
                    for bbox, text, conf in ocr_results:
                        if conf > 0.1:
                            clean_text = self.clean_text_advanced(text)
                            if len(clean_text) >= 3:
                                results.append((clean_text, conf * 0.9, 'easyocr'))
                except:
                    continue
        
        # Tesseract with multiple PSM modes
        if TESSERACT_AVAILABLE:
            psm_modes = [8, 7, 6, 13]
            for variant in variants[:2]:
                for psm in psm_modes:
                    try:
                        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        text = pytesseract.image_to_string(variant, config=config).strip()
                        clean_text = self.clean_text_advanced(text)
                        if len(clean_text) >= 3:
                            results.append((clean_text, 0.7, f'tesseract_psm{psm}'))
                    except:
                        continue
        
        # Score and select best result
        best_result = None
        best_score = 0
        
        for text, conf, method in results:
            score = self.calculate_text_score(text, conf)
            if score > best_score:
                best_score = score
                best_result = {
                    'text': text,
                    'confidence': conf,
                    'method': method,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                }
        
        return best_result if best_score > 0.2 else None
    
    def create_image_variants(self, image):
        """Create multiple processed versions"""
        variants = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if small
        h, w = gray.shape
        if h < 40 or w < 80:
            scale = max(40/h, 80/w, 2.0)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Variant 1: CLAHE + OTSU
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh1)
        
        # Variant 2: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive)
        
        # Variant 3: Inverted
        inverted = cv2.bitwise_not(thresh1)
        variants.append(inverted)
        
        # Variant 4: Sharpened
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(sharp_thresh)
        
        return variants
    
    def clean_text_advanced(self, text):
        """Advanced text cleaning"""
        if not text:
            return ""
        
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        # Smart character corrections
        corrections = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1', '|': '1',
            'S': '5', 'Z': '2', 'G': '6', 'B': '8'
        }
        
        corrected = list(text)
        for i, char in enumerate(text):
            # Apply corrections based on position
            if i >= len(text) - 3:  # Numbers more likely at end
                if char in corrections and corrections[char].isdigit():
                    corrected[i] = corrections[char]
            elif i < 3:  # Letters more likely at start
                if char.isdigit() and char in ['0', '1', '5']:
                    reverse_map = {'0': 'O', '1': 'I', '5': 'S'}
                    corrected[i] = reverse_map.get(char, char)
        
        return ''.join(corrected)
    
    def calculate_text_score(self, text, confidence):
        """Calculate quality score for recognized text"""
        if not text or len(text) < 4 or len(text) > 10:
            return 0
        
        score = confidence * 0.5  # Reduce base confidence weight
        
        # Reject obvious garbage
        if len(set(text)) < 3:  # Too few unique characters
            return 0
        if text.count('0') > len(text) * 0.6:  # Too many zeros
            return 0
        if re.search(r'[A-Z]{6,}|[0-9]{6,}', text):  # Too many consecutive letters/numbers
            return 0
        
        # Length scoring
        if 5 <= len(text) <= 8:
            score += 0.4
        elif len(text) == 4:
            score += 0.2
        else:
            score -= 0.2
        
        # Pattern matching (strict)
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        if not (has_letter and has_number):
            return 0
        
        # Valid patterns only
        if re.match(r'^[A-Z]{2,3}[0-9]{3,4}$', text):  # AB123, ABC1234
            score += 0.6
        elif re.match(r'^[0-9]{3,4}[A-Z]{2,3}$', text):  # 123AB, 1234ABC
            score += 0.5
        elif re.match(r'^[A-Z][0-9]{3}[A-Z]{2}$', text):  # A123BC
            score += 0.4
        else:
            score -= 0.3  # Penalize unusual patterns
        
        return max(0, min(score, 1.0))
    
    def boxes_overlap(self, box1, box2, threshold=0.5):
        """Check if two bounding boxes overlap"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union > threshold
    
    def process_vehicle_for_plates(self, image, vehicle_bbox, vehicle_id):
        """Simplified processing with immediate results"""
        results = []
        
        try:
            plates = self.detect_plates(image, vehicle_bbox)
            
            for plate in plates:
                recognition_result = self.recognize_text_advanced(plate['image'])
                
                if recognition_result and recognition_result.get('text'):
                    plate_text = recognition_result['text']
                    score = recognition_result.get('score', 0)
                    
                    if score > 0.2 and len(plate_text) >= 3:  # Very low threshold
                        result = {
                            'vehicle_id': vehicle_id,
                            'vehicle_bbox': vehicle_bbox,
                            'plate_bbox': plate['bbox'],
                            'plate_text': plate_text,
                            'confidence': recognition_result['confidence'],
                            'method': recognition_result.get('method', 'unknown'),
                            'score': score,
                            'timestamp': recognition_result['timestamp']
                        }
                        
                        results.append(result)
                        self.detection_history.append(result)
                        
                        if len(self.detection_history) > 20:
                            self.detection_history = self.detection_history[-20:]
                        
                        print(f"🎯 Plate detected: {plate_text} (score: {score:.2f})")
                        break
        
        except Exception as e:
            print(f"❌ Error in processing: {e}")
        
        return results
    
    def select_best_reading(self, vehicle_id):
        """Select the most correct reading from multiple attempts"""
        if vehicle_id not in self.vehicle_readings or not self.vehicle_readings[vehicle_id]:
            return None
        
        readings = self.vehicle_readings[vehicle_id]
        
        # If only one reading, return it
        if len(readings) == 1:
            return readings[0]
        
        # Find most common reading
        text_counts = {}
        for reading in readings:
            text = reading['text']
            if text not in text_counts:
                text_counts[text] = []
            text_counts[text].append(reading)
        
        # Select based on frequency and quality
        best_text = None
        best_score = 0
        
        for text, text_readings in text_counts.items():
            # Calculate combined score: frequency + average quality
            frequency_score = len(text_readings) / len(readings)
            avg_quality = sum(r['score'] for r in text_readings) / len(text_readings)
            combined_score = (frequency_score * 0.6) + (avg_quality * 0.4)
            
            if combined_score > best_score:
                best_score = combined_score
                best_text = text
        
        if best_text:
            # Return the highest quality reading of the best text
            best_readings = text_counts[best_text]
            return max(best_readings, key=lambda x: x['score'])
        
        return None
    
    def get_recent_detections(self, limit=10):
        """Get recent detections"""
        return self.detection_history[-limit:] if self.detection_history else []
    
    def draw_plate_detections(self, image, detections):
        """Draw detections on image with better visibility"""
        for detection in detections:
            px1, py1, px2, py2 = detection['plate_bbox']
            
            # Draw plate box
            cv2.rectangle(image, (px1, py1), (px2, py2), (0, 255, 255), 3)
            
            # Draw license plate text with background
            plate_text = detection['plate_text']
            text = f"PLATE: {plate_text}"
            
            # Text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(image, (px1, py1 - 35), (px1 + text_size[0] + 10, py1 - 5), (0, 255, 255), -1)
            
            # Text
            cv2.putText(image, text, (px1 + 5, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return image