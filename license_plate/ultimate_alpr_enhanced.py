"""
Enhanced License Plate Detection and Recognition Module
Incorporates features from ultimateALPR-SDK for improved performance and accuracy
"""

import cv2
import numpy as np
import re
import os
from datetime import datetime
import logging
from collections import deque

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

# Try to import our multi-engine OCR system
try:
    from license_plate.multi_engine_ocr import multi_engine_ocr
    MULTI_OCR_AVAILABLE = True
except ImportError:
    MULTI_OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class UltimateEnhancedLicensePlateDetector:
    def __init__(self):
        self.plate_model = None
        self.ocr_reader = None
        self.detection_history = deque(maxlen=100)
        self.load_models()
        
        # Enhanced license plate patterns for different regions
        self.plate_patterns = {
            'US': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[A-Z]{2}[0-9]{5}$',  # AB12345
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[A-Z][0-9]{3}[A-Z]{2}$',  # A123BC
            ],
            'EU': [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$',  # AB12CDE
                r'^[A-Z]{1}[0-9]{3}[A-Z]{3}$', # A123BCD
                r'^[A-Z]{1,3}[0-9]{2,4}[A-Z]{1,2}$',  # Various EU formats
            ],
            'INDIA': [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # MH12AB1234
                r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',  # MH12A1234
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',  # 12AB1234
            ],
            'INTERNATIONAL': [
                r'^[A-Z0-9]{3,8}$',  # Generic international pattern
            ]
        }
        
        # Performance optimization settings
        self.pyramidal_search_enabled = True
        self.pyramidal_search_sensitivity = 0.28
        self.detect_minscore = 0.1
        self.recogn_minscore = 0.3
        
        # UltimateALPR-SDK inspired features
        self.klass_lpci_enabled = True  # License Plate Country Identification
        self.klass_vcr_enabled = True   # Vehicle Color Recognition
        self.klass_vmmr_enabled = True  # Vehicle Make Model Recognition
        self.recogn_rectify_enabled = False
        self.recogn_score_type = "min"
        
        # Multi-charset support
        self.charsets = ['latin', 'international']
        
        # Enhanced preprocessing
        self.image_enhancement_enabled = True
        
        # Vehicle characteristics for better plate location
        self.vehicle_plate_positions = {
            'car': {'y_ratio': 0.7, 'height_ratio': 0.3},
            'truck': {'y_ratio': 0.6, 'height_ratio': 0.4},
            'bus': {'y_ratio': 0.6, 'height_ratio': 0.4},
            'motorcycle': {'y_ratio': 0.5, 'height_ratio': 0.3}
        }
    
    def load_models(self):
        """Load license plate detection and OCR models with enhanced error handling"""
        try:
            if YOLO_AVAILABLE:
                logger.info("🔧 Loading enhanced license plate detection model...")
                try:
                    # Try custom license plate model first
                    model_path = 'yolov8_license_plate2 (1).pt'
                    if os.path.exists(model_path):
                        self.plate_model = YOLO(model_path)
                        logger.info("✅ Custom license plate model loaded")
                    else:
                        # Fallback to general object detection with optimized settings
                        model_path = 'yolov8n.pt'
                        if os.path.exists(model_path):
                            self.plate_model = YOLO(model_path)
                            logger.info("✅ Using general YOLO model for plate detection")
                        else:
                            logger.warning("⚠️ No YOLO model found")
                            self.plate_model = None
                except Exception as e:
                    logger.error(f"Error loading YOLO model: {e}")
                    self.plate_model = None
            
            # Load OCR reader with fallback chain
            if MULTI_OCR_AVAILABLE:
                logger.info("✅ Using multi-engine OCR system")
            elif OCR_AVAILABLE:
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
    
    def detect_license_plates(self, image, vehicle_bbox=None, vehicle_type='car'):
        """
        Enhanced license plate detection with pyramidal search and multi-scale detection
        Incorporates features from ultimateALPR-SDK for improved performance
        Args:
            image: Input image
            vehicle_bbox: Optional vehicle bounding box to focus search
            vehicle_type: Type of vehicle for better plate location prediction
        Returns:
            List of detected license plate regions
        """
        plates = []
        
        if self.plate_model is None:
            return self.enhanced_fallback_plate_detection(image, vehicle_bbox, vehicle_type)
        
        try:
            # Determine search region based on vehicle type
            search_region, offset = self._get_search_region(image, vehicle_bbox, vehicle_type)
            
            # Apply image enhancement if enabled
            if self.image_enhancement_enabled:
                search_region = self._enhance_image_for_detection(search_region)
            
            # Multi-scale detection for better accuracy
            scales = [1.0, 0.8, 1.2] if self.pyramidal_search_enabled else [1.0]
            all_detections = []
            
            for scale in scales:
                if scale != 1.0:
                    scaled_region = cv2.resize(search_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                else:
                    scaled_region = search_region
                
                # Ensure image has 3 channels for YOLO
                if len(scaled_region.shape) == 2:  # Grayscale
                    scaled_region_rgb = cv2.cvtColor(scaled_region, cv2.COLOR_GRAY2RGB)
                elif scaled_region.shape[2] == 1:  # Single channel
                    scaled_region_rgb = cv2.cvtColor(scaled_region, cv2.COLOR_GRAY2RGB)
                else:
                    scaled_region_rgb = scaled_region
                
                # Run YOLO detection with optimized parameters
                results = self.plate_model(
                    scaled_region_rgb, 
                    conf=self.detect_minscore,
                    classes=[0] if 'license_plate' in str(self.plate_model.model) else None
                )
                
                # Adjust coordinates back to original scale and image
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            px1, py1, px2, py2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            
                            # Adjust for scale
                            if scale != 1.0:
                                px1, py1, px2, py2 = int(px1/scale), int(py1/scale), int(px2/scale), int(py2/scale)
                            
                            # Adjust for search region offset
                            px1 += offset[0]
                            py1 += offset[1]
                            px2 += offset[0]
                            py2 += offset[1]
                            
                            # Ensure bounds are valid
                            px1, py1 = max(0, px1), max(0, py1)
                            px2, py2 = min(image.shape[1], px2), min(image.shape[0], py2)
                            
                            if px2 > px1 and py2 > py1:
                                plate_image = image[py1:py2, px1:px2]
                                # Enhanced plate analysis
                                plate_info = self._analyze_plate_characteristics(plate_image)
                                
                                all_detections.append({
                                    'bbox': (px1, py1, px2, py2),
                                    'confidence': confidence,
                                    'image': plate_image,
                                    'characteristics': plate_info
                                })
            
            # Remove duplicates and select best candidates
            plates = self._remove_duplicate_plates(all_detections)
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced plate detection: {e}")
            return self.enhanced_fallback_plate_detection(image, vehicle_bbox, vehicle_type)
        
        return plates
    
    def _get_search_region(self, image, vehicle_bbox, vehicle_type):
        """Get optimized search region based on vehicle type"""
        if vehicle_bbox is not None:
            x1, y1, x2, y2 = vehicle_bbox
            
            # Use vehicle type to predict plate location
            if vehicle_type in self.vehicle_plate_positions:
                pos_info = self.vehicle_plate_positions[vehicle_type]
                search_y1 = y1 + int((y2 - y1) * pos_info['y_ratio'])
                search_height = int((y2 - y1) * pos_info['height_ratio'])
                search_y2 = min(y2, search_y1 + search_height)
            else:
                # Default to bottom third
                mid_y = (y1 + y2) // 2
                search_y1 = mid_y
                search_y2 = y2
            
            search_region = image[search_y1:search_y2, x1:x2]
            offset = (x1, search_y1)
        else:
            search_region = image
            offset = (0, 0)
        
        return search_region, offset
    
    def _analyze_plate_characteristics(self, plate_image):
        """
        Analyze license plate characteristics for better recognition
        Inspired by ultimateALPR-SDK features
        """
        if plate_image is None or plate_image.size == 0:
            return {}
        
        characteristics = {
            'aspect_ratio': 0.0,
            'area': 0,
            'color_info': {},
            'texture_info': {},
            'country_likelihood': {}
        }
        
        try:
            h, w = plate_image.shape[:2]
            characteristics['aspect_ratio'] = w / h if h > 0 else 0
            characteristics['area'] = w * h
            
            # Color analysis for country identification
            if self.klass_lpci_enabled:
                characteristics['country_likelihood'] = self._estimate_plate_country(plate_image)
            
            # Color information
            if self.klass_vcr_enabled:
                characteristics['color_info'] = self._analyze_plate_color(plate_image)
                
        except Exception as e:
            logger.debug(f"Plate characteristics analysis failed: {e}")
        
        return characteristics
    
    def _remove_duplicate_plates(self, plates):
        """Remove duplicate plate detections and return unique ones"""
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
        
        # Sort by confidence and return top candidates
        unique_plates.sort(key=lambda x: x['confidence'], reverse=True)
        return unique_plates[:5]  # Return top 5 candidates
    
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
    
    def enhanced_fallback_plate_detection(self, image, vehicle_bbox=None, vehicle_type='car'):
        """
        Enhanced fallback license plate detection using advanced CV methods
        """
        plates = []
        
        try:
            # Get search region based on vehicle type
            search_region, offset = self._get_search_region(image, vehicle_bbox, vehicle_type)
            
            # Advanced preprocessing for better detection
            enhanced_gray = self._enhance_image_for_detection(search_region)
            
            # Multiple detection approaches with different parameters
            potential_plates = []
            
            # Method 1: Edge-based detection with adaptive parameters
            edge_plates = self._edge_based_detection(enhanced_gray, sensitivity=0.3)
            potential_plates.extend([(x, y, w, h, 0.4) for x, y, w, h in edge_plates])
            
            # Method 2: Morphological operations with different kernels
            morph_plates = self._morphological_detection(enhanced_gray)
            potential_plates.extend([(x, y, w, h, 0.6) for x, y, w, h in morph_plates])
            
            # Method 3: Contour-based detection with shape analysis
            contour_plates = self._contour_based_detection(enhanced_gray)
            potential_plates.extend([(x, y, w, h, 0.5) for x, y, w, h in contour_plates])
            
            # Remove duplicates and select best candidates
            unique_plates = self._remove_duplicate_candidates(potential_plates)
            
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
            logger.error(f"❌ Error in enhanced fallback detection: {e}")
        
        return plates
    
    def _enhance_image_for_detection(self, image):
        """Enhance image for better plate detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for better contrast (with error handling)
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
        except cv2.error:
            # Fallback if CLAHE fails
            enhanced = gray
        
        # Bilateral filter to reduce noise while preserving edges
        try:
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        except cv2.error:
            filtered = enhanced
        
        return filtered
    
    def _estimate_plate_country(self, plate_image):
        """
        Estimate license plate country based on visual characteristics
        Inspired by ultimateALPR-SDK's License Plate Country Identification (LPCI)
        """
        country_likelihood = {}
        
        try:
            # Convert to different color spaces for analysis
            if len(plate_image.shape) == 3:
                hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image
                hsv = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
                hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
            
            # Analyze aspect ratio (different countries have different plate formats)
            h, w = plate_image.shape[:2]
            aspect_ratio = w / h if h > 0 else 0
            
            # Initialize likelihoods
            country_likelihood = {
                'india': 0.0,
                'usa': 0.0,
                'eu': 0.0,
                'unknown': 0.5
            }
            
            # India: Typically wider than tall (aspect ratio > 2.0)
            if 1.8 <= aspect_ratio <= 5.0:
                country_likelihood['india'] += 0.3
            
            # USA: More rectangular (aspect ratio ~ 2.0)
            if 1.5 <= aspect_ratio <= 2.5:
                country_likelihood['usa'] += 0.3
            
            # EU: Various formats but often with specific color patterns
            if 2.0 <= aspect_ratio <= 4.0:
                country_likelihood['eu'] += 0.2
            
            # Color analysis for blue strips (EU plates often have blue strips)
            if len(plate_image.shape) == 3:
                # Check for blue regions
                blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
                blue_pixels = cv2.countNonZero(blue_mask)
                total_pixels = h * w
                blue_ratio = blue_pixels / total_pixels if total_pixels > 0 else 0
                
                if 0.05 <= blue_ratio <= 0.3:  # EU plates often have blue strips
                    country_likelihood['eu'] += 0.2
            
            # Normalize likelihoods
            total = sum(country_likelihood.values())
            if total > 0:
                for country in country_likelihood:
                    country_likelihood[country] /= total
            
        except Exception as e:
            logger.debug(f"Country estimation failed: {e}")
        
        return country_likelihood
    
    def _analyze_plate_color(self, plate_image):
        """
        Analyze plate color characteristics
        Inspired by ultimateALPR-SDK's Vehicle Color Recognition (VCR)
        """
        color_info = {
            'dominant_color': 'unknown',
            'colorfulness': 0.0
        }
        
        try:
            if len(plate_image.shape) == 3:
                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
                
                # Calculate colorfulness metric
                (h, s, v) = cv2.split(hsv.astype("float"))
                colorfulness = np.std(s) + 0.3 * np.mean(s)
                color_info['colorfulness'] = colorfulness
                
                # Find dominant color
                height, width = plate_image.shape[:2]
                if height > 0 and width > 0:
                    # Sample colors from different regions
                    colors = []
                    for i in range(0, height, max(1, height//5)):
                        for j in range(0, width, max(1, width//5)):
                            pixel = hsv[i, j]
                            colors.append(pixel)
                    
                    if colors:
                        # Average hue value
                        avg_hue = np.mean([c[0] for c in colors])
                        
                        # Map hue to color name
                        if 0 <= avg_hue < 15 or 165 <= avg_hue <= 180:
                            color_info['dominant_color'] = 'red'
                        elif 15 <= avg_hue < 45:
                            color_info['dominant_color'] = 'yellow'
                        elif 45 <= avg_hue < 75:
                            color_info['dominant_color'] = 'green'
                        elif 75 <= avg_hue < 105:
                            color_info['dominant_color'] = 'cyan'
                        elif 105 <= avg_hue < 135:
                            color_info['dominant_color'] = 'blue'
                        elif 135 <= avg_hue < 165:
                            color_info['dominant_color'] = 'purple'
                        else:
                            color_info['dominant_color'] = 'white' if np.mean(v) > 200 else 'black'
            
        except Exception as e:
            logger.debug(f"Plate color analysis failed: {e}")
        
        return color_info
    
    def _edge_based_detection(self, gray, sensitivity=0.3):
        """Advanced edge-based plate detection"""
        plates = []
        
        # Multiple edge detection approaches
        for low_thresh in [30, 50, 100]:
            edges = cv2.Canny(gray, low_thresh, low_thresh * 2)
            
            # Morphological operations to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 20000:  # Adjusted area range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # License plates typically have aspect ratios between 2:1 to 5:1
                    if 1.5 < aspect_ratio < 8.0:
                        # Additional shape filtering
                        rect_area = w * h
                        extent = float(area) / rect_area
                        
                        # Filter out too sparse or too dense regions
                        if 0.2 <= extent <= 0.9:
                            plates.append((x, y, w, h))
        
        return plates
    
    def _morphological_detection(self, gray):
        """Morphological operations for plate detection"""
        plates = []
        
        # Try different kernel sizes
        kernel_sizes = [(20, 5), (25, 8), (30, 10)]
        
        for kernel_w, kernel_h in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Threshold
            _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 300 < area < 25000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 2.0 < aspect_ratio < 6.0:
                        plates.append((x, y, w, h))
        
        return plates
    
    def _contour_based_detection(self, gray):
        """Contour-based detection with shape analysis"""
        plates = []
        
        # Find contours with different retrieval modes
        for mode in [cv2.RETR_EXTERNAL, cv2.RETR_TREE]:
            contours, _ = cv2.findContours(gray, mode, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 250 < area < 18000:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Check aspect ratio
                    if 1.8 < aspect_ratio < 7.0:
                        # Additional checks for plate-like shapes
                        perimeter = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                        
                        # Plates are typically rectangles (4 sides)
                        if len(approx) >= 4:
                            plates.append((x, y, w, h))
        
        return plates
    
    def _remove_duplicate_candidates(self, candidates):
        """Remove duplicate plate candidates"""
        if len(candidates) <= 1:
            return candidates
        
        unique_candidates = []
        for x, y, w, h, conf in candidates:
            is_duplicate = False
            for ux, uy, uw, uh, _ in unique_candidates:
                # Calculate overlap
                x_overlap = max(0, min(x + w, ux + uw) - max(x, ux))
                y_overlap = max(0, min(y + h, uy + uh) - max(y, uy))
                overlap_area = x_overlap * y_overlap
                
                # Calculate union area
                area1 = w * h
                area2 = uw * uh
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0:
                    overlap_ratio = overlap_area / union_area
                    if overlap_ratio > 0.5:  # 50% overlap threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_candidates.append((x, y, w, h, conf))
        
        # Sort by confidence
        unique_candidates.sort(key=lambda x: x[4], reverse=True)
        return unique_candidates
    
    def recognize_text(self, plate_image, plate_characteristics=None):
        """
        Enhanced text recognition using multi-engine OCR with fallback
        Incorporates features from ultimateALPR-SDK for improved accuracy
        """
        if plate_image is None or plate_image.size == 0:
            return None
        
        try:
            # Use multi-engine OCR if available
            if MULTI_OCR_AVAILABLE:
                # Preprocess multiple variants for better OCR
                variants = self._create_ocr_variants(plate_image)
                best_result = None
                best_score = 0
                
                for variant in variants:
                    try:
                        result = multi_engine_ocr.recognize_text(variant)
                        if result:
                            # Score based on confidence and text validity
                            score = result['confidence'] * self._score_plate_text(result['text'])
                            
                            # Adjust score based on plate characteristics if available
                            if plate_characteristics:
                                # Boost score for plates with characteristics matching known formats
                                country_likelihood = plate_characteristics.get('country_likelihood', {})
                                if country_likelihood:
                                    max_likelihood = max(country_likelihood.values())
                                    score *= (1.0 + max_likelihood)  # Boost by up to 2x
                            
                            if score > best_score and score > self.recogn_minscore:
                                best_score = score
                                best_result = result
                    except Exception as e:
                        logger.debug(f"OCR variant failed: {e}")
                        continue
                
                if best_result:
                    # Add enhanced metadata
                    best_result['enhanced'] = True
                    if plate_characteristics:
                        best_result['plate_characteristics'] = plate_characteristics
                    return best_result
            
            # Fallback to individual OCR engines
            result = self._fallback_ocr_recognition(plate_image)
            if result:
                result['enhanced'] = False
                if plate_characteristics:
                    result['plate_characteristics'] = plate_characteristics
                return result
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced text recognition: {e}")
            
        return None
    
    def _create_ocr_variants(self, plate_image):
        """Create multiple image variants for better OCR results"""
        variants = []
        
        # Original image
        variants.append(plate_image)
        
        # Preprocess the image for better OCR
        processed = self._preprocess_plate_image(plate_image)
        variants.append(processed)
        
        # Inverted version (for dark plates on light background)
        if len(processed.shape) == 2:  # Grayscale
            inverted = cv2.bitwise_not(processed)
            variants.append(inverted)
        
        # Resized versions for different scales
        height, width = plate_image.shape[:2]
        if height > 0 and width > 0:
            for scale in [0.8, 1.2, 1.5]:
                new_width = int(width * scale)
                new_height = int(height * scale)
                if new_width > 20 and new_height > 10:  # Minimum size
                    resized = cv2.resize(plate_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    variants.append(resized)
        
        return variants[:8]  # Limit to 8 variants to avoid overprocessing
    
    def _fallback_ocr_recognition(self, plate_image):
        """Fallback OCR recognition using individual engines"""
        # Preprocess the image
        processed_image = self._preprocess_plate_image(plate_image)
        
        text = None
        confidence = 0.0
        
        # Try EasyOCR first
        if self.ocr_reader is not None:
            try:
                results = self.ocr_reader.readtext(processed_image, detail=1)
                if results:
                    # Get the result with highest confidence
                    best_result = max(results, key=lambda x: x[2])
                    bbox, detected_text, conf = best_result
                    text = detected_text
                    confidence = conf
            except Exception as e:
                logger.error(f"EasyOCR error: {e}")
        
        # Fallback to Tesseract if EasyOCR fails or low confidence
        if (text is None or confidence < 0.5) and TESSERACT_AVAILABLE:
            try:
                # Try multiple configurations
                configs = [
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    r'--oem 3 --psm 6'
                ]
                
                best_tess_result = None
                best_tess_conf = 0
                
                for config in configs:
                    try:
                        tess_text = pytesseract.image_to_string(processed_image, config=config).strip()
                        if tess_text:
                            # Estimate confidence (Tesseract doesn't provide it directly)
                            tess_conf = 0.6  # Default confidence
                            if len(tess_text) >= 4 and tess_text.isalnum():
                                tess_conf = 0.7
                            
                            if tess_conf > best_tess_conf:
                                best_tess_conf = tess_conf
                                best_tess_result = tess_text
                    except:
                        continue
                
                if best_tess_result and best_tess_conf > confidence:
                    text = best_tess_result
                    confidence = best_tess_conf
                    
            except Exception as e:
                logger.error(f"Tesseract error: {e}")
        
        # Clean and validate the text
        if text:
            text = self._clean_plate_text(text)
            if self._validate_plate_text(text):
                return {
                    'text': text,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
        
        return None
    
    def _preprocess_plate_image(self, image):
        """
        Advanced preprocessing for license plate images
        """
        try:
            # Resize image if too small
            height, width = image.shape[:2]
            if height < 30 or width < 60:
                scale_factor = max(30/height, 60/width, 2.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            try:
                enhanced = clahe.apply(gray)
            except cv2.error:
                # Fallback if CLAHE fails
                enhanced = gray
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Try multiple thresholding approaches
            # Method 1: OTSU
            try:
                _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except:
                _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            
            # Method 2: Adaptive threshold
            try:
                thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            except:
                thresh2 = thresh1
            
            # Method 3: Simple threshold
            _, thresh3 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            
            # Choose the best threshold based on text area
            thresholds = [thresh1, thresh2, thresh3]
            best_thresh = thresh1
            max_text_area = 0
            
            for thresh in thresholds:
                try:
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    text_area = sum(cv2.contourArea(c) for c in contours if 5 < cv2.contourArea(c) < 500)
                    if text_area > max_text_area:
                        max_text_area = text_area
                        best_thresh = thresh
                except:
                    continue
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            try:
                cleaned = cv2.morphologyEx(best_thresh, cv2.MORPH_CLOSE, kernel)
            except:
                cleaned = best_thresh
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error in advanced preprocessing: {e}")
            return image
    
    def _clean_plate_text(self, text):
        """
        Enhanced cleaning and formatting of license plate text
        """
        if not text:
            return ""
        
        # Remove unwanted characters and convert to uppercase
        text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        # Smart OCR mistake correction based on common patterns
        corrections = {
            'O': '0', 'Q': '0', 'D': '0', 'C': '0',
            'I': '1', 'L': '1', '|': '1', 'J': '1',
            'Z': '2', 'S': '5', 'G': '6', 'B': '8'
        }
        
        # Apply corrections with position awareness
        corrected = list(text)
        for i, char in enumerate(text):
            if char in corrections:
                # In positions more likely to be numbers (end of plate)
                if i >= len(text) - 3:
                    corrected[i] = corrections[char]
                # In positions more likely to be letters (beginning of plate)
                elif i < 3 and char in ['0', '1', '2', '5', '6', '8']:
                    # Don't change numbers at the beginning that might be correct
                    pass
                else:
                    corrected[i] = corrections[char]
        
        return ''.join(corrected)
    
    def _score_plate_text(self, text):
        """Score plate text based on likelihood of being valid"""
        if not text or len(text) < 3:
            return 0.0
        
        score = 0.0
        
        # Length bonus (ideal length for plates is 5-10 characters)
        if 5 <= len(text) <= 10:
            score += 0.3
        elif 3 <= len(text) <= 12:
            score += 0.1
        
        # Character mix bonus (plates typically have both letters and numbers)
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        if has_letter and has_number:
            score += 0.4
        elif text.isalnum():
            score += 0.2
        
        # Pattern matching bonus
        for region, patterns in self.plate_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text):
                    score += 0.3
                    break
        
        return min(score, 1.0)
    
    def _validate_plate_text(self, text):
        """Enhanced validation for license plate text"""
        if not text or len(text) < 3 or len(text) > 12:
            return False
        
        # Must be alphanumeric
        if not text.isalnum():
            return False
        
        # Check against regional patterns
        for region, patterns in self.plate_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text):
                    return True
        
        # Fallback: accept any alphanumeric text with reasonable length
        # that has both letters and numbers or is all digits with minimum length
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        return (has_letter and has_number) or (text.isdigit() and len(text) >= 4)
    
    def process_vehicle_for_plates(self, image, vehicle_bbox, vehicle_id, vehicle_type='car'):
        """
        Enhanced processing of a detected vehicle to find and recognize license plates
        Incorporates features from ultimateALPR-SDK for improved performance and accuracy
        """
        results = []
        
        try:
            # Use enhanced detection with vehicle type awareness
            plates = self.detect_license_plates(image, vehicle_bbox, vehicle_type)
            
            # If no plates found, try fallback detection
            if not plates:
                plates = self.enhanced_fallback_plate_detection(image, vehicle_bbox, vehicle_type)
            
            # Process each detected plate
            for plate in plates:
                # Enhanced OCR with multiple attempts and plate characteristics
                plate_characteristics = plate.get('characteristics', {})
                recognition_result = self.recognize_text(plate['image'], plate_characteristics)
                
                if recognition_result and recognition_result.get('text'):
                    result = {
                        'vehicle_id': vehicle_id,
                        'vehicle_bbox': vehicle_bbox,
                        'plate_bbox': plate['bbox'],
                        'plate_text': recognition_result['text'],
                        'confidence': recognition_result['confidence'],
                        'detection_confidence': plate['confidence'],
                        'timestamp': recognition_result['timestamp'],
                        'enhanced': recognition_result.get('enhanced', False),
                        'plate_characteristics': recognition_result.get('plate_characteristics', {})
                    }
                    
                    # Add country estimation if available
                    country_likelihood = plate_characteristics.get('country_likelihood', {})
                    if country_likelihood:
                        result['estimated_country'] = max(country_likelihood, key=country_likelihood.get)
                    
                    results.append(result)
                    
                    # Add to history
                    self.detection_history.append(result)
                    
                    logger.info(f"🎯 License plate detected: {recognition_result['text']} (confidence: {recognition_result['confidence']:.2f})")
        
        except Exception as e:
            logger.error(f"❌ Error processing vehicle for plates: {e}")
        
        return results
    
    def get_recent_detections(self, limit=10):
        """
        Get recent license plate detections
        """
        return list(self.detection_history)[-limit:] if self.detection_history else []
    
    def draw_plate_detections(self, image, detections):
        """
        Enhanced drawing of license plate detections on the image
        """
        for detection in detections:
            # Draw plate bounding box
            px1, py1, px2, py2 = detection['plate_bbox']
            cv2.rectangle(image, (px1, py1), (px2, py2), (0, 255, 255), 2)  # Cyan box
            
            # Draw plate text with better formatting
            text = f"{detection['plate_text']} ({detection['confidence']:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Background for text
            cv2.rectangle(image, (px1, py1 - text_size[1] - 10), 
                         (px1 + text_size[0], py1), (0, 255, 255), -1)
            
            # Text
            cv2.putText(image, text, (px1, py1 - 5), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Draw detection confidence as secondary info
            det_text = f"Det: {detection['detection_confidence']:.2f}"
            det_text_size = cv2.getTextSize(det_text, font, 0.4, 1)[0]
            cv2.rectangle(image, (px1, py2), 
                         (px1 + det_text_size[0], py2 + det_text_size[1] + 5), (0, 255, 255), -1)
            cv2.putText(image, det_text, (px1, py2 + det_text_size[1]), 
                       font, 0.4, (0, 0, 0), 1)
        
        return image