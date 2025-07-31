"""
License Plate Detection and Recognition Module
Combines YOLO for plate detection with OCR for text recognition
"""

import cv2
import numpy as np
import re
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
                # Try to load a license plate detection model
                # You can train your own or use a pre-trained one
                logger.info("🔧 Loading license plate detection model...")
                try:
                    # Try custom license plate model first
                    self.plate_model = YOLO('license_plate.pt')
                    logger.info("✅ Custom license plate model loaded")
                except:
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
            
            # Run YOLO detection
            results = self.plate_model(cropped_image, conf=0.3)
            
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
        Fallback license plate detection using traditional CV methods
        """
        plates = []
        
        try:
            # Focus on vehicle region if provided
            if vehicle_bbox is not None:
                x1, y1, x2, y2 = vehicle_bbox
                search_region = image[y1:y2, x1:x2]
                offset = (x1, y1)
            else:
                search_region = image
                offset = (0, 0)
            
            # Convert to grayscale
            gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Find edges
            edges = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that could be license plates
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 10000:  # Filter by area
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (license plates are typically wider than tall)
                aspect_ratio = w / h
                if aspect_ratio < 2.0 or aspect_ratio > 6.0:
                    continue
                
                # Adjust coordinates
                px1, py1 = x + offset[0], y + offset[1]
                px2, py2 = px1 + w, py1 + h
                
                plates.append({
                    'bbox': (px1, py1, px2, py2),
                    'confidence': 0.5,  # Default confidence for fallback method
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
            
            # Try EasyOCR first
            if self.ocr_reader is not None:
                try:
                    results = self.ocr_reader.readtext(processed_image)
                    if results:
                        # Combine all detected text
                        text_parts = []
                        confidences = []
                        for (bbox, detected_text, conf) in results:
                            if conf > 0.3:  # Minimum confidence threshold
                                text_parts.append(detected_text.strip())
                                confidences.append(conf)
                        
                        if text_parts:
                            text = ''.join(text_parts).upper()
                            confidence = sum(confidences) / len(confidences)
                
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
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
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
        
        # Remove common OCR mistakes
        text = text.replace('O', '0')  # Replace O with 0 in certain contexts
        text = text.replace('I', '1')  # Replace I with 1 in certain contexts
        text = text.replace('S', '5')  # Replace S with 5 in certain contexts
        
        return text
    
    def validate_plate_text(self, text):
        """
        Validate if the text matches common license plate patterns
        """
        if not text or len(text) < 4 or len(text) > 10:
            return False
        
        # Check against known patterns
        for region, patterns in self.plate_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text):
                    return True
        
        # Basic validation: should contain both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return has_letter and has_number
    
    def process_vehicle_for_plates(self, image, vehicle_bbox, vehicle_id):
        """
        Process a detected vehicle to find and recognize license plates
        """
        results = []
        
        try:
            # Detect license plates in the vehicle region
            plates = self.detect_license_plates(image, vehicle_bbox)
            
            for plate in plates:
                # Recognize text from the plate
                recognition_result = self.recognize_text(plate['image'])
                
                if recognition_result:
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
                    
                    logger.info(f"🎯 License plate detected: {recognition_result['text']} (confidence: {recognition_result['confidence']:.2f})")
        
        except Exception as e:
            logger.error(f"❌ Error processing vehicle for plates: {e}")
        
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