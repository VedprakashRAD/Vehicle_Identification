"""
Intelligent License Plate Detector with advanced preprocessing and multiple OCR engines
"""

import cv2
import numpy as np
import re
import os
from datetime import datetime

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

class IntelligentLicensePlateDetector:
    def __init__(self):
        self.ocr_reader = None
        self.detection_history = []
        self.load_ocr()
        
    def load_ocr(self):
        if OCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("✅ EasyOCR loaded")
            except:
                pass
    
    def detect_license_plates(self, image, vehicle_bbox):
        """Detect license plate regions using multiple CV techniques"""
        x1, y1, x2, y2 = vehicle_bbox
        
        # Focus on bottom 40% of vehicle where plates are located
        roi_y1 = y1 + int((y2 - y1) * 0.6)
        roi = image[roi_y1:y2, x1:x2]
        
        if roi.size == 0:
            return []
        
        plates = []
        
        # Method 1: Contour detection with aspect ratio filtering
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 5000:  # Reasonable plate area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plates are typically 2:1 to 5:1 ratio
                if 2.0 <= aspect_ratio <= 5.0 and w > 60 and h > 15:
                    # Adjust coordinates back to full image
                    plate_bbox = (x1 + x, roi_y1 + y, x1 + x + w, roi_y1 + y + h)
                    plate_img = image[roi_y1 + y:roi_y1 + y + h, x1 + x:x1 + x + w]
                    
                    plates.append({
                        'bbox': plate_bbox,
                        'image': plate_img,
                        'confidence': 0.7
                    })
        
        # Method 2: Morphological operations for rectangular detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 400 < area < 8000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 2.5 <= aspect_ratio <= 4.5 and w > 80 and h > 20:
                    plate_bbox = (x1 + x, roi_y1 + y, x1 + x + w, roi_y1 + y + h)
                    plate_img = image[roi_y1 + y:roi_y1 + y + h, x1 + x:x1 + x + w]
                    
                    # Check if this overlaps with existing plates
                    overlap = False
                    for existing in plates:
                        if self.calculate_overlap(plate_bbox, existing['bbox']) > 0.5:
                            overlap = True
                            break
                    
                    if not overlap:
                        plates.append({
                            'bbox': plate_bbox,
                            'image': plate_img,
                            'confidence': 0.8
                        })
        
        return plates[:3]  # Return top 3 candidates
    
    def calculate_overlap(self, box1, box2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def preprocess_for_ocr(self, image):
        """Advanced preprocessing for better OCR results"""
        if image.size == 0:
            return []
        
        variants = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small
        h, w = gray.shape
        if h < 30 or w < 60:
            scale = max(30/h, 60/w, 2.0)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Variant 1: CLAHE + Gaussian blur + OTSU
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh1)
        
        # Variant 2: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive)
        
        # Variant 3: Inverted (for dark plates)
        inverted = cv2.bitwise_not(thresh1)
        variants.append(inverted)
        
        # Variant 4: Bilateral filter + threshold
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh2)
        
        return variants
    
    def recognize_text(self, plate_image):
        """Multi-engine OCR with intelligent text validation"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        variants = self.preprocess_for_ocr(plate_image)
        candidates = []
        
        # Try EasyOCR on all variants
        if self.ocr_reader:
            for variant in variants:
                try:
                    results = self.ocr_reader.readtext(variant, detail=1, paragraph=False, width_ths=0.7)
                    for bbox, text, conf in results:
                        if conf > 0.1:
                            clean_text = self.clean_text(text)
                            if self.is_valid_plate(clean_text):
                                candidates.append((clean_text, conf * 0.9, 'easyocr'))
                except:
                    continue
        
        # Try Tesseract with different configurations
        if TESSERACT_AVAILABLE:
            configs = [
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            for variant in variants[:2]:  # Limit variants for speed
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(variant, config=config).strip()
                        clean_text = self.clean_text(text)
                        if self.is_valid_plate(clean_text):
                            candidates.append((clean_text, 0.6, 'tesseract'))
                    except:
                        continue
        
        # Select best candidate
        if not candidates:
            return None
        
        # Score candidates
        best_candidate = None
        best_score = 0
        
        for text, conf, method in candidates:
            score = self.score_plate_text(text, conf)
            if score > best_score:
                best_score = score
                best_candidate = {
                    'text': text,
                    'confidence': conf,
                    'method': method,
                    'timestamp': datetime.now().isoformat()
                }
        
        return best_candidate if best_score > 0.3 else None
    
    def clean_text(self, text):
        """Clean and normalize OCR text"""
        if not text:
            return ""
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        # Common OCR corrections
        corrections = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1', '|': '1',
            'S': '5', 'Z': '2', 'G': '6', 'B': '8'
        }
        
        # Apply corrections intelligently based on position
        corrected = list(text)
        for i, char in enumerate(text):
            # Numbers more likely at end
            if i >= len(text) - 3:
                if char in corrections and corrections[char].isdigit():
                    corrected[i] = corrections[char]
        
        return ''.join(corrected)
    
    def is_valid_plate(self, text):
        """Check if text looks like a valid license plate"""
        if not text or len(text) < 3 or len(text) > 10:
            return False
        
        # Must be alphanumeric
        if not text.isalnum():
            return False
        
        # Should have both letters and numbers (most common)
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        # Accept if has both letters and numbers, or if all numbers (some plates)
        return (has_letter and has_number) or (text.isdigit() and len(text) >= 4)
    
    def score_plate_text(self, text, confidence):
        """Score the quality of detected plate text"""
        if not text:
            return 0
        
        score = confidence * 0.7
        
        # Length bonus
        if 4 <= len(text) <= 8:
            score += 0.3
        elif len(text) == 3 or len(text) == 9:
            score += 0.1
        
        # Pattern bonus
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        if has_letter and has_number:
            score += 0.4
        
        # Common patterns
        if re.match(r'^[A-Z]{2,3}[0-9]{3,4}$', text):  # AB123, ABC1234
            score += 0.5
        elif re.match(r'^[0-9]{3,4}[A-Z]{2,3}$', text):  # 123AB
            score += 0.4
        elif re.match(r'^[A-Z][0-9]{3}[A-Z]{2}$', text):  # A123BC
            score += 0.3
        
        return min(score, 1.0)
    
    def process_vehicle_for_plates(self, image, vehicle_bbox, vehicle_id):
        """Main processing function"""
        results = []
        
        try:
            plates = self.detect_license_plates(image, vehicle_bbox)
            
            for plate in plates:
                recognition_result = self.recognize_text(plate['image'])
                
                if recognition_result:
                    result = {
                        'vehicle_id': vehicle_id,
                        'vehicle_bbox': vehicle_bbox,
                        'plate_bbox': plate['bbox'],
                        'plate_text': recognition_result['text'],
                        'confidence': recognition_result['confidence'],
                        'method': recognition_result.get('method', 'unknown'),
                        'timestamp': recognition_result['timestamp']
                    }
                    
                    results.append(result)
                    self.detection_history.append(result)
                    
                    if len(self.detection_history) > 50:
                        self.detection_history = self.detection_history[-50:]
                    
                    print(f"🎯 Plate detected: {recognition_result['text']} (conf: {recognition_result['confidence']:.2f}, method: {recognition_result.get('method')})")
                    break  # Take first good detection
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return results
    
    def get_recent_detections(self, limit=10):
        return self.detection_history[-limit:] if self.detection_history else []
    
    def draw_plate_detections(self, image, detections):
        for detection in detections:
            px1, py1, px2, py2 = detection['plate_bbox']
            cv2.rectangle(image, (px1, py1), (px2, py2), (0, 255, 255), 2)
            
            text = f"PLATE: {detection['plate_text']}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(image, (px1, py1 - 30), (px1 + text_size[0] + 10, py1), (0, 255, 255), -1)
            cv2.putText(image, text, (px1 + 5, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return image