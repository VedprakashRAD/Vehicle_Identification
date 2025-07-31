"""
License Plate Detection and Recognition Module
Combines YOLO for plate detection with OCR for text recognition
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

logger = logging.getLogger(__name__)

class LicensePlateDetector:
    def __init__(self):
        self.plate_model = None
        self.ocr_reader = None
        self.detection_history = []
        self.load_models()
        
        # License plate patterns for different regions
        self.plate_patterns = {
            'US': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[A-Z]{2}[0-9]{5}$',  # AB12345
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
            ],
            'EU': [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$',  # AB12CDE
                r'^[A-Z]{1}[0-9]{3}[A-Z]{3}$', # A123BCD
            ],
            'INDIA': [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # MH12AB1234
                r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',  # MH12A1234
            ]
        }
    
    def load_models(self):
        """Load license plate detection and OCR models"""
        try:
            if YOLO_AVAILABLE:
                logger.info("🔧 Loading license plate detection model...")
                try:
                    # Try custom license plate model first
                    if os.path.exists('yolov8_license_plate2 (1).pt'):
                        self.plate_model = YOLO('yolov8_license_plate2 (1).pt')
                        logger.info("✅ Custom license plate model loaded")
                    else:
                        # Fallback to general object detection
                        self.plate_model = YOLO('yolov8n.pt')
                        logger.info("✅ Using general YOLO model for plate detection")
                except Exception as e:
                    logger.error(f"Error loading YOLO model: {e}")
                    # Fallback to general object detection
                    self.plate_model = YOLO('yolov8n.pt')
                    logger.info("✅ Using general YOLO model for plate detection")
            
            # Load OCR reader
            if OCR_AVAILABLE:
                logger.info("🔧 Loading EasyOCR...")
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("✅ EasyOCR loaded successfully")
            elif TESSERACT_AVAILABLE:
                logger.info("✅ Using Tesseract OCR")
            else:
                logger.warning("⚠️ No OCR engine available")
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.plate_model = None
            self.ocr_reader = None
    
    def detect_license_plates(self, image, vehicle_bbox=None):
        """
        Detect license plates in the image
        Args:
            image: Input image
            vehicle_bbox: Optional vehicle bounding box to focus search
        Returns:
            List of detected license plate regions
        """
        plates = []
        
        if self.plate_model is None:
            return self.fallback_plate_detection(image, vehicle_bbox)
        
        try:
            # If vehicle bbox is provided, crop the region
            if vehicle_bbox is not None:
                x1, y1, x2, y2 = vehicle_bbox
                # Expand the bbox slightly to ensure we don't miss plates
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.shape[1], x2 + margin)
                y2 = min(image.shape[0], y2 + margin)
                
                cropped_image = image[y1:y2, x1:x2]
                offset = (x1, y1)
            else:
                cropped_image = image
                offset = (0, 0)
            
            # Run YOLO detection with lower confidence for better detection
            results = self.plate_model(cropped_image, conf=0.2, classes=[0] if 'license_plate' in str(self.plate_model.model) else None)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Adjust coordinates if we used cropped image
                        px1 += offset[0]
                        py1 += offset[1]
                        px2 += offset[0]
                        py2 += offset[1]
                        
                        plates.append({
                            'bbox': (px1, py1, px2, py2),
                            'confidence': confidence,
                            'image': image[py1:py2, px1:px2]
                        })
            
        except Exception as e:
            logger.error(f"❌ Error in plate detection: {e}")
            return self.fallback_plate_detection(image, vehicle_bbox)
        
        return plates
    
    def fallback_plate_detection(self, image, vehicle_bbox=None):
        """
        Enhanced fallback license plate detection using traditional CV methods
        """
        plates = []
        
        try:
            # Focus on vehicle region if provided
            if vehicle_bbox is not None:
                x1, y1, x2, y2 = vehicle_bbox
                # Focus on lower half of vehicle where plates are typically located
                mid_y = (y1 + y2) // 2
                search_region = image[mid_y:y2, x1:x2]
                offset = (x1, mid_y)
            else:
                search_region = image
                offset = (0, 0)
            
            # Convert to grayscale
            gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
            
            # Multiple detection approaches
            potential_plates = []
            
            # Method 1: Edge-based detection
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(filtered, 30, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 300 < area < 15000:  # Broader area range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 1.5 < aspect_ratio < 8.0:  # Broader aspect ratio
                        potential_plates.append((x, y, w, h, 0.4))
            
            # Method 2: Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 400 < area < 12000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 2.0 < aspect_ratio < 6.0:
                        potential_plates.append((x, y, w, h, 0.6))
            
            # Method 3: Template matching for rectangular regions
            template_sizes = [(100, 30), (120, 40), (150, 50)]
            for tw, th in template_sizes:
                if tw < search_region.shape[1] and th < search_region.shape[0]:
                    # Create a simple rectangular template
                    template = np.ones((th, tw), dtype=np.uint8) * 255
                    cv2.rectangle(template, (2, 2), (tw-3, th-3), 0, 2)
                    
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= 0.3)
                    
                    for pt in zip(*locations[::-1]):
                        potential_plates.append((pt[0], pt[1], tw, th, 0.3))
            
            # Remove duplicates and select best candidates
            unique_plates = []
            for x, y, w, h, conf in potential_plates:
                # Check if this overlaps significantly with existing plates
                overlap = False
                for ux, uy, uw, uh, _ in unique_plates:
                    if (abs(x - ux) < w//2 and abs(y - uy) < h//2):
                        overlap = True
                        break
                
                if not overlap:
                    unique_plates.append((x, y, w, h, conf))
            
            # Convert to final format
            for x, y, w, h, conf in unique_plates[:3]:  # Limit to top 3 candidates
                px1, py1 = x + offset[0], y + offset[1]
                px2, py2 = px1 + w, py1 + h
                
                # Ensure coordinates are within image bounds
                px1 = max(0, px1)
                py1 = max(0, py1)
                px2 = min(image.shape[1], px2)
                py2 = min(image.shape[0], py2)
                
                if px2 > px1 and py2 > py1:
                    plates.append({
                        'bbox': (px1, py1, px2, py2),
                        'confidence': conf,
                        'image': image[py1:py2, px1:px2]
                    })
            
        except Exception as e:
            logger.error(f"❌ Error in fallback detection: {e}")
        
        return plates
    
    def recognize_text(self, plate_image):
        """
        Recognize text from license plate image using OCR
        """
        if plate_image is None or plate_image.size == 0:
            return None
        
        try:
            # Preprocess the image for better OCR
            processed_image = self.preprocess_plate_image(plate_image)
            
            text = None
            confidence = 0.0
            
            # Try EasyOCR first with multiple image versions
            if self.ocr_reader is not None:
                try:
                    # Try OCR on multiple versions of the image
                    images_to_try = [processed_image, plate_image]
                    
                    # Also try with different scaling
                    if processed_image.shape[0] < 100:
                        scaled = cv2.resize(processed_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        images_to_try.append(scaled)
                    
                    best_result = None
                    best_confidence = 0
                    
                    for img in images_to_try:
                        try:
                            results = self.ocr_reader.readtext(img)
                            if results:
                                # Combine all detected text
                                text_parts = []
                                confidences = []
                                for (bbox, detected_text, conf) in results:
                                    if conf > 0.2:  # Lower threshold for better detection
                                        clean_text = re.sub(r'[^A-Z0-9]', '', detected_text.upper())
                                        if len(clean_text) >= 3:  # Minimum length
                                            text_parts.append(clean_text)
                                            confidences.append(conf)
                                
                                if text_parts and confidences:
                                    combined_text = ''.join(text_parts)
                                    avg_confidence = sum(confidences) / len(confidences)
                                    
                                    if avg_confidence > best_confidence:
                                        best_confidence = avg_confidence
                                        best_result = (combined_text, avg_confidence)
                        except:
                            continue
                    
                    if best_result:
                        text, confidence = best_result
                
                except Exception as e:
                    logger.error(f"EasyOCR error: {e}")
            
            # Fallback to Tesseract if EasyOCR fails
            if (text is None or confidence < 0.5) and TESSERACT_AVAILABLE:
                try:
                    # Configure Tesseract for license plates
                    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(processed_image, config=custom_config).strip().upper()
                    confidence = 0.6  # Default confidence for Tesseract
                except Exception as e:
                    logger.error(f"Tesseract error: {e}")
            
            # Clean and validate the text
            if text:
                text = self.clean_plate_text(text)
                if self.validate_plate_text(text):
                    return {
                        'text': text,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    }
            
        except Exception as e:
            logger.error(f"❌ Error in text recognition: {e}")
        
        return None
    
    def preprocess_plate_image(self, image):
        """
        Preprocess license plate image for better OCR results
        """
        try:
            # Resize image if too small
            height, width = image.shape[:2]
            if height < 50 or width < 100:
                scale_factor = max(50/height, 100/width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Try multiple thresholding approaches
            # Method 1: OTSU
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 2: Adaptive threshold
            thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Method 3: Simple threshold
            _, thresh3 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            
            # Choose the best threshold based on text area
            thresholds = [thresh1, thresh2, thresh3]
            best_thresh = thresh1
            max_text_area = 0
            
            for thresh in thresholds:
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                text_area = sum(cv2.contourArea(c) for c in contours if 10 < cv2.contourArea(c) < 1000)
                if text_area > max_text_area:
                    max_text_area = text_area
                    best_thresh = thresh
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            cleaned = cv2.morphologyEx(best_thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return image
    
    def clean_plate_text(self, text):
        """
        Clean and format license plate text
        """
        if not text:
            return ""
        
        # Remove unwanted characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Smart OCR mistake correction based on position
        # Only replace in likely numeric positions
        if len(text) >= 4:
            # For patterns like ABC123, replace in likely number positions
            corrected = list(text)
            for i, char in enumerate(text):
                if i >= len(text) - 4:  # Last 4 characters more likely to be numbers
                    if char == 'O': corrected[i] = '0'
                    elif char == 'I': corrected[i] = '1'
                    elif char == 'S': corrected[i] = '5'
                    elif char == 'Z': corrected[i] = '2'
            text = ''.join(corrected)
        
        return text
    
    def validate_plate_text(self, text):
        """
        More lenient validation for license plate text
        """
        if not text or len(text) < 3 or len(text) > 12:
            return False
        
        # Accept any alphanumeric text with reasonable length
        return text.isalnum()
    
    def process_vehicle_for_plates(self, image, vehicle_bbox, vehicle_id):
        """
        Process a detected vehicle to find and recognize license plates
        """
        results = []
        
        try:
            # Always try fallback detection first (more reliable)
            plates = self.fallback_plate_detection(image, vehicle_bbox)
            
            # If no plates found, try YOLO detection
            if not plates and self.plate_model is not None:
                plates = self.detect_license_plates(image, vehicle_bbox)
            
            # If still no plates, create synthetic plate regions for OCR testing
            if not plates:
                x1, y1, x2, y2 = vehicle_bbox
                # Try bottom portion of vehicle
                bottom_region = image[int(y2-50):y2, x1:x2]
                if bottom_region.size > 0:
                    plates = [{
                        'bbox': (x1, int(y2-50), x2, y2),
                        'confidence': 0.3,
                        'image': bottom_region
                    }]
            
            for plate in plates:
                # Always try OCR even on low confidence detections
                recognition_result = self.recognize_text(plate['image'])
                
                if recognition_result and recognition_result.get('text'):
                    result = {
                        'vehicle_id': vehicle_id,
                        'vehicle_bbox': vehicle_bbox,
                        'plate_bbox': plate['bbox'],
                        'plate_text': recognition_result['text'],
                        'confidence': recognition_result['confidence'],
                        'detection_confidence': plate['confidence'],
                        'timestamp': recognition_result['timestamp']
                    }
                    
                    results.append(result)
                    
                    # Add to history
                    self.detection_history.append(result)
                    
                    # Keep only recent detections (last 100)
                    if len(self.detection_history) > 100:
                        self.detection_history = self.detection_history[-100:]
                    
                    print(f"🎯 License plate detected: {recognition_result['text']} (confidence: {recognition_result['confidence']:.2f})")
        
        except Exception as e:
            print(f"❌ Error processing vehicle for plates: {e}")
        
        return results
    
    def get_recent_detections(self, limit=10):
        """
        Get recent license plate detections
        """
        return self.detection_history[-limit:] if self.detection_history else []
    
    def draw_plate_detections(self, image, detections):
        """
        Draw license plate detections on the image
        """
        for detection in detections:
            # Draw plate bounding box
            px1, py1, px2, py2 = detection['plate_bbox']
            cv2.rectangle(image, (px1, py1), (px2, py2), (255, 255, 0), 2)  # Yellow box
            
            # Draw plate text
            text = f"{detection['plate_text']} ({detection['confidence']:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(image, (px1, py1 - text_size[1] - 10), 
                         (px1 + text_size[0], py1), (255, 255, 0), -1)
            
            # Text
            cv2.putText(image, text, (px1, py1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image