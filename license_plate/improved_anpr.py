"""
Improved ANPR System optimized for Indian license plates
Based on research from PlateRecognizePy and other YOLOv8+EasyOCR implementations
"""

import cv2
import numpy as np
import re
import os
from datetime import datetime
from collections import deque
import logging

# Import our multi-engine OCR system
try:
    from license_plate.multi_engine_ocr import multi_engine_ocr
    OCR_AVAILABLE = True
except ImportError:
    # Fallback to individual OCR engines
    try:
        import easyocr
        OCR_AVAILABLE = True
    except ImportError:
        OCR_AVAILABLE = False
        print("EasyOCR not available")

    try:
        import pytesseract
        TESSERACT_AVAILABLE = True
    except ImportError:
        TESSERACT_AVAILABLE = False
        print("Tesseract OCR not available")

logger = logging.getLogger(__name__)

class ImprovedANPR:
    def __init__(self):
        self.ocr_reader = None
        self.detection_history = deque(maxlen=100)
        self.load_ocr()
        
        # Indian state codes for validation
        self.indian_state_codes = {
            'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DN', 'DD', 'DL', 'GA', 'GJ', 'HR', 
            'HP', 'JK', 'JH', 'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 
            'PY', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB'
        }
        
    def load_ocr(self):
        """Load OCR engines with fallback options"""
        if OCR_AVAILABLE:
            try:
                # Load EasyOCR with GPU support if available
                self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("✅ EasyOCR loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load EasyOCR: {e}")
                self.ocr_reader = None
        else:
            logger.warning("EasyOCR not available")
            
        if not self.ocr_reader and TESSERACT_AVAILABLE:
            try:
                # Test Tesseract availability
                pytesseract.get_tesseract_version()
                logger.info("✅ Tesseract OCR available as fallback")
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")
    
    def detect_license_plates(self, image, vehicle_bbox):
        """
        Detect license plate regions using improved techniques
        Optimized for Indian number plates
        """
        if image is None or image.size == 0:
            return []
            
        x1, y1, x2, y2 = vehicle_bbox
        
        # Focus on bottom portion of vehicle where plates are typically located
        # For Indian vehicles, plates are often in the lower third
        roi_y1 = y1 + int((y2 - y1) * 0.65)
        roi_y2 = y1 + int((y2 - y1) * 0.95)
        roi = image[roi_y1:roi_y2, x1:x2]
        
        if roi.size == 0:
            return []
        
        plates = []
        
        # Multi-scale detection for different plate sizes
        scales = [1.0, 0.8, 1.2]
        for scale in scales:
            if scale != 1.0:
                scaled_roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            else:
                scaled_roi = roi
                
            detected_plates = self._detect_plates_in_roi(scaled_roi, x1, roi_y1, scale)
            plates.extend(detected_plates)
        
        # Remove duplicates and sort by confidence
        unique_plates = self._remove_duplicate_plates(plates)
        unique_plates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_plates[:3]  # Return top 3 candidates
    
    def _detect_plates_in_roi(self, roi, offset_x, offset_y, scale):
        """Detect plates in a region of interest"""
        plates = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast in Indian conditions
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Multiple preprocessing approaches
        preprocessings = [
            ('canny', self._canny_detection),
            ('morph', self._morphological_detection),
            ('threshold', self._threshold_detection)
        ]
        
        for method_name, method_func in preprocessings:
            try:
                detected = method_func(enhanced, roi)
                for plate in detected:
                    # Adjust coordinates back to full image
                    px1, py1, px2, py2 = plate['bbox']
                    adjusted_bbox = (
                        int(offset_x + px1 / scale),
                        int(offset_y + py1 / scale),
                        int(offset_x + px2 / scale),
                        int(offset_y + py2 / scale)
                    )
                    
                    plate_img = self._extract_plate_image(roi, plate['bbox'])
                    if plate_img is not None and plate_img.size > 0:
                        plates.append({
                            'bbox': adjusted_bbox,
                            'image': plate_img,
                            'confidence': plate['confidence'] * 0.9,  # Adjust for scale
                            'method': method_name
                        })
            except Exception as e:
                logger.debug(f"Detection method {method_name} failed: {e}")
                continue
                
        return plates
    
    def _canny_detection(self, gray, original):
        """Detect plates using Canny edge detection"""
        plates = []
        
        # Edge detection with different thresholds
        for low_thresh in [50, 100, 150]:
            edges = cv2.Canny(gray, low_thresh, low_thresh * 2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Indian plates typically have area between 500-10000 pixels
                if 300 < area < 15000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Indian plates have aspect ratios typically between 2:1 to 5:1
                    if 1.5 <= aspect_ratio <= 6.0 and w > 50 and h > 15:
                        # Additional shape filtering
                        rect_area = w * h
                        extent = float(area) / rect_area
                        
                        if 0.2 <= extent <= 0.9:  # Filter out too sparse or too dense regions
                            plates.append({
                                'bbox': (x, y, x + w, y + h),
                                'confidence': min(0.8, area / 5000.0),  # Normalize confidence
                                'area': area,
                                'aspect_ratio': aspect_ratio
                            })
        
        return plates
    
    def _morphological_detection(self, gray, original):
        """Detect plates using morphological operations"""
        plates = []
        
        # Try different kernel sizes
        for kernel_size in [(15, 5), (20, 8), (25, 10)]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Threshold
            _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 400 < area < 20000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 2.0 <= aspect_ratio <= 5.5 and w > 60 and h > 20:
                        plates.append({
                            'bbox': (x, y, x + w, y + h),
                            'confidence': min(0.75, area / 10000.0),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
        
        return plates
    
    def _threshold_detection(self, gray, original):
        """Detect plates using adaptive thresholding"""
        plates = []
        
        # Try different adaptive threshold methods
        methods = [
            (cv2.ADAPTIVE_THRESH_MEAN_C, 11),
            (cv2.ADAPTIVE_THRESH_MEAN_C, 15),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15)
        ]
        
        for method, block_size in methods:
            try:
                thresh = cv2.adaptiveThreshold(
                    gray, 255, method, cv2.THRESH_BINARY, block_size, 2
                )
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 300 < area < 15000:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h
                        
                        if 1.8 <= aspect_ratio <= 6.0 and w > 50 and h > 15:
                            plates.append({
                                'bbox': (x, y, x + w, y + h),
                                'confidence': min(0.7, area / 8000.0),
                                'area': area,
                                'aspect_ratio': aspect_ratio
                            })
            except Exception as e:
                continue
                
        return plates
    
    def _extract_plate_image(self, roi, bbox):
        """Extract plate image from ROI"""
        x1, y1, x2, y2 = bbox
        h, w = roi.shape[:2]
        
        # Ensure bounds are valid
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        if x2 > x1 and y2 > y1:
            return roi[y1:y2, x1:x2]
        return None
    
    def _remove_duplicate_plates(self, plates):
        """Remove duplicate plate detections"""
        if len(plates) <= 1:
            return plates
            
        unique_plates = []
        for plate in plates:
            is_duplicate = False
            for existing in unique_plates:
                overlap = self._calculate_overlap(plate['bbox'], existing['bbox'])
                if overlap > 0.6:  # 60% overlap threshold
                    # Keep the one with higher confidence
                    if plate['confidence'] > existing['confidence']:
                        existing.update(plate)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_plates.append(plate)
                
        return unique_plates
    
    def _calculate_overlap(self, box1, box2):
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
        """Advanced preprocessing for better OCR results on Indian plates"""
        if image is None or image.size == 0:
            return []
        
        variants = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small for better OCR
        h, w = gray.shape
        if h < 40 or w < 80:
            scale = max(60/h, 100/w, 2.0)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Variant 1: Standard preprocessing (works well for white plates)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced1 = clahe.apply(gray)
        blurred1 = cv2.GaussianBlur(enhanced1, (3, 3), 0)
        _, thresh1 = cv2.threshold(blurred1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(('white_plate', thresh1))
        
        # Variant 2: Inverted (works well for yellow/dark plates)
        inverted1 = cv2.bitwise_not(thresh1)
        variants.append(('yellow_plate', inverted1))
        
        # Variant 3: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        variants.append(('adaptive', adaptive))
        
        # Variant 4: Inverted adaptive
        inverted_adaptive = cv2.bitwise_not(adaptive)
        variants.append(('inverted_adaptive', inverted_adaptive))
        
        # Variant 5: Bilateral filter + threshold (good for noisy images)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(('bilateral', thresh2))
        
        # Variant 6: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        _, thresh3 = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(('morph', thresh3))
        
        return variants
    
    def recognize_text(self, plate_image):
        """Multi-engine OCR with intelligent text validation for Indian plates"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Use our multi-engine OCR system
        if OCR_AVAILABLE:
            try:
                # Preprocess image variants for better OCR
                variants = self.preprocess_for_ocr(plate_image)
                candidates = []
                
                # Try OCR on all variants
                for variant_name, variant in variants[:4]:  # Limit to top variants
                    try:
                        result = multi_engine_ocr.recognize_text(variant)
                        if result and self.is_valid_indian_plate(result['text']):
                            # Adjust confidence based on variant quality
                            adjusted_confidence = result['confidence'] * 0.9
                            result['confidence'] = adjusted_confidence
                            candidates.append(result)
                    except Exception as e:
                        logger.debug(f"OCR failed on {variant_name}: {e}")
                        continue
                
                # Select best candidate
                if candidates:
                    # Score candidates based on Indian plate patterns
                    best_candidate = None
                    best_score = 0
                    
                    for candidate in candidates:
                        score = self.score_indian_plate_text(candidate['text'], candidate['confidence'])
                        if score > best_score:
                            best_score = score
                            best_candidate = candidate
                    
                    if best_score > 0.4:
                        return best_candidate
                        
            except Exception as e:
                logger.error(f"Multi-engine OCR failed: {e}")
        
        # Fallback to original method if multi-engine OCR fails
        return self._fallback_recognize_text(plate_image)
    
    def _fallback_recognize_text(self, plate_image):
        """Fallback OCR method using individual engines"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        variants = self.preprocess_for_ocr(plate_image)
        candidates = []
        
        # Try EasyOCR on all variants
        if self.ocr_reader:
            for variant_name, variant in variants:
                try:
                    # Adjust parameters for Indian plates
                    results = self.ocr_reader.readtext(
                        variant, 
                        detail=1, 
                        paragraph=False, 
                        width_ths=0.7,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        min_size=10
                    )
                    
                    for bbox, text, conf in results:
                        if conf > 0.1:  # Lower threshold to catch more candidates
                            clean_text = self.clean_indian_plate_text(text)
                            if self.is_valid_indian_plate(clean_text):
                                candidates.append({
                                    'text': clean_text,
                                    'confidence': conf * 0.9,  # Adjust for preprocessing
                                    'method': f'easyocr_{variant_name}',
                                    'timestamp': datetime.now().isoformat()
                                })
                except Exception as e:
                    logger.debug(f"EasyOCR failed on {variant_name}: {e}")
                    continue
        
        # Try Tesseract with Indian plate specific configurations
        if TESSERACT_AVAILABLE:
            configs = [
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            for variant_name, variant in variants[:3]:  # Limit variants for speed
                for i, config in enumerate(configs):
                    try:
                        text = pytesseract.image_to_string(variant, config=config).strip()
                        clean_text = self.clean_indian_plate_text(text)
                        if self.is_valid_indian_plate(clean_text):
                            candidates.append({
                                'text': clean_text,
                                'confidence': 0.6,  # Standard confidence for Tesseract
                                'method': f'tesseract_{variant_name}_{i}',
                                'timestamp': datetime.now().isoformat()
                            })
                    except Exception as e:
                        logger.debug(f"Tesseract failed on {variant_name}: {e}")
                        continue
        
        # Select best candidate
        if not candidates:
            return None
        
        # Score candidates based on Indian plate patterns
        best_candidate = None
        best_score = 0
        
        for candidate in candidates:
            score = self.score_indian_plate_text(candidate['text'], candidate['confidence'])
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate if best_score > 0.4 else None
    
    def clean_indian_plate_text(self, text):
        """Clean and normalize OCR text for Indian plates"""
        if not text:
            return ""
        
        # Remove spaces and non-alphanumeric characters
        text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        # Common OCR corrections for Indian plates
        corrections = {
            'O': '0', 'Q': '0', 'D': '0', 'C': '0',
            'I': '1', 'L': '1', '|': '1', 'J': '1',
            'Z': '2', 'S': '5', 'G': '6', 'B': '8'
        }
        
        # Apply corrections
        corrected = list(text)
        for i, char in enumerate(text):
            if char in corrections:
                corrected[i] = corrections[char]
        
        return ''.join(corrected)
    
    def is_valid_indian_plate(self, text):
        """Check if text looks like a valid Indian license plate"""
        if not text or len(text) < 3 or len(text) > 12:
            return False
        
        # Must be alphanumeric
        if not text.isalnum():
            return False
        
        # Check for common Indian plate patterns
        patterns = [
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$',  # DL01AB1234, MH12AB1234
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$',  # DL1C1234, MH12ABC1234
            r'^[A-Z]{3}[0-9]{4}$',                         # ABC1234
            r'^[A-Z]{2}[0-9]{5}$',                         # DL12345
            r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',                 # 12AB1234
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        # Check if it starts with a valid state code
        if len(text) >= 2 and text[:2] in self.indian_state_codes:
            has_letter = any(c.isalpha() for c in text)
            has_number = any(c.isdigit() for c in text)
            return has_letter and has_number
        
        # Accept if has both letters and numbers (general case)
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        return (has_letter and has_number) or (text.isdigit() and len(text) >= 4)
    
    def score_indian_plate_text(self, text, confidence):
        """Score the quality of detected Indian plate text"""
        if not text:
            return 0
        
        score = confidence * 0.7
        
        # Length bonus (Indian plates are typically 6-10 characters)
        if 6 <= len(text) <= 10:
            score += 0.3
        elif len(text) == 5 or len(text) == 11:
            score += 0.1
        
        # Pattern bonus for common Indian formats
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        if has_letter and has_number:
            score += 0.4
        
        # Specific pattern bonuses
        if re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$', text):  # DL01AB1234
            score += 0.5
        elif re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$', text):  # DL1C1234
            score += 0.4
        elif re.match(r'^[0-9]{2}[A-Z]{2}[0-9]{4}$', text):  # 12AB1234
            score += 0.3
        elif re.match(r'^[A-Z]{3}[0-9]{4}$', text):  # ABC1234
            score += 0.2
        
        # State code bonus
        if len(text) >= 2 and text[:2] in self.indian_state_codes:
            score += 0.3
        
        return min(score, 1.0)
    
    def process_vehicle_for_plates(self, image, vehicle_bbox, vehicle_id):
        """Main processing function for vehicle plate detection"""
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
                        'method': recognition_result['method'],
                        'timestamp': recognition_result['timestamp']
                    }
                    
                    results.append(result)
                    self.detection_history.append(result)
                    
                    logger.info(f"🎯 Indian Plate detected: {recognition_result['text']} (conf: {recognition_result['confidence']:.2f}, method: {recognition_result['method']})")
                    break  # Take first good detection (highest confidence)
        
        except Exception as e:
            logger.error(f"❌ Error in plate processing: {e}")
        
        return results
    
    def get_recent_detections(self, limit=10):
        """Get recent plate detections"""
        return list(self.detection_history)[-limit:] if self.detection_history else []
    
    def draw_plate_detections(self, image, detections):
        """Draw plate detections on image"""
        for detection in detections:
            px1, py1, px2, py2 = detection['plate_bbox']
            cv2.rectangle(image, (px1, py1), (px2, py2), (0, 255, 255), 2)
            
            text = f"{detection['plate_text']} ({detection['confidence']:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (px1, py1 - 25), (px1 + text_size[0] + 10, py1), (0, 255, 255), -1)
            cv2.putText(image, text, (px1 + 5, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image