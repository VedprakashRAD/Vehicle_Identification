"""
Multi-engine OCR system with PaddleOCR as primary, EasyOCR and Tesseract as fallbacks
"""

import cv2
import numpy as np
import re
import logging
from datetime import datetime

# Try to import all OCR engines
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    print("PaddleOCR not available")

try:
    import easyocr
    EASY_OCR_AVAILABLE = True
except ImportError:
    EASY_OCR_AVAILABLE = False
    print("EasyOCR not available")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract OCR not available")

logger = logging.getLogger(__name__)

class MultiEngineOCR:
    def __init__(self):
        self.paddle_ocr = None
        self.easy_ocr = None
        self.tesseract_available = TESSERACT_AVAILABLE
        
        # Load OCR engines in order of preference
        self._load_ocr_engines()
        
    def _load_ocr_engines(self):
        """Load OCR engines with proper error handling"""
        # Load PaddleOCR (primary)
        if PADDLE_OCR_AVAILABLE:
            try:
                # Initialize PaddleOCR with optimized settings for license plates
                # Try different parameter combinations based on PaddleOCR version
                try:
                    self.paddle_ocr = PaddleOCR(
                        use_angle_cls=False,
                        lang='en',
                        use_gpu=False  # Set to True if you have GPU support
                    )
                except TypeError:
                    # Fallback if some parameters are not supported
                    try:
                        self.paddle_ocr = PaddleOCR(
                            use_angle_cls=False,
                            lang='en'
                        )
                    except:
                        # Minimal initialization
                        self.paddle_ocr = PaddleOCR(lang='en')
                logger.info("✅ PaddleOCR loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load PaddleOCR: {e}")
                self.paddle_ocr = None
        else:
            logger.warning("PaddleOCR not available")
            
        # Load EasyOCR (secondary)
        if EASY_OCR_AVAILABLE:
            try:
                self.easy_ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("✅ EasyOCR loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load EasyOCR: {e}")
                self.easy_ocr = None
        else:
            logger.warning("EasyOCR not available")
            
        # Check Tesseract availability
        if TESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                logger.info("✅ Tesseract OCR available as fallback")
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")
                self.tesseract_available = False
        else:
            logger.warning("Tesseract OCR not available")
    
    def recognize_text(self, image):
        """
        Recognize text using multiple OCR engines in order of preference:
        1. PaddleOCR (primary)
        2. EasyOCR (secondary)
        3. Tesseract OCR (fallback)
        """
        if image is None or image.size == 0:
            return None
            
        candidates = []
        
        # Try PaddleOCR first
        if self.paddle_ocr:
            try:
                result = self._paddle_ocr_recognize(image)
                if result and self._is_valid_plate_text(result['text']):
                    candidates.append(result)
                    logger.debug(f"PaddleOCR result: {result['text']} (confidence: {result['confidence']:.2f})")
            except Exception as e:
                logger.debug(f"PaddleOCR failed: {e}")
        
        # Try EasyOCR if PaddleOCR failed or not available
        if not candidates and self.easy_ocr:
            try:
                result = self._easy_ocr_recognize(image)
                if result and self._is_valid_plate_text(result['text']):
                    candidates.append(result)
                    logger.debug(f"EasyOCR result: {result['text']} (confidence: {result['confidence']:.2f})")
            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")
        
        # Try Tesseract as last resort
        if not candidates and self.tesseract_available:
            try:
                result = self._tesseract_recognize(image)
                if result and self._is_valid_plate_text(result['text']):
                    candidates.append(result)
                    logger.debug(f"Tesseract result: {result['text']} (confidence: {result['confidence']:.2f})")
            except Exception as e:
                logger.debug(f"Tesseract failed: {e}")
        
        # Return best candidate
        if candidates:
            # Sort by confidence
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            return candidates[0]
        
        return None
    
    def _paddle_ocr_recognize(self, image):
        """Recognize text using PaddleOCR"""
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)
        
        # Run PaddleOCR
        result = self.paddle_ocr.ocr(processed_image, det=False, rec=True)
        
        if result and result[0]:
            # PaddleOCR returns list of lists
            text_results = result[0]
            if text_results:
                # Get the result with highest confidence
                best_result = max(text_results, key=lambda x: x[1][1])
                text, (confidence, _) = best_result[1]
                
                return {
                    'text': self._clean_text(text),
                    'confidence': confidence,
                    'method': 'paddleocr',
                    'timestamp': datetime.now().isoformat()
                }
        
        return None
    
    def _easy_ocr_recognize(self, image):
        """Recognize text using EasyOCR"""
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)
        
        # Run EasyOCR
        results = self.easy_ocr.readtext(processed_image, detail=1, paragraph=False)
        
        if results:
            # Get the result with highest confidence
            best_result = max(results, key=lambda x: x[2])
            bbox, text, confidence = best_result
            
            return {
                'text': self._clean_text(text),
                'confidence': confidence,
                'method': 'easyocr',
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _tesseract_recognize(self, image):
        """Recognize text using Tesseract OCR"""
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)
        
        # Try different configurations
        configs = [
            '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]
        
        best_result = None
        best_confidence = 0
        
        for config in configs:
            try:
                # Get detailed data including confidence
                data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
                
                # Find the word with highest confidence
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 0:  # Only consider words with positive confidence
                        text = data['text'][i].strip()
                        confidence = int(data['conf'][i]) / 100.0  # Convert to 0-1 scale
                        
                        if confidence > best_confidence and len(text) > 2:
                            best_confidence = confidence
                            best_result = {
                                'text': self._clean_text(text),
                                'confidence': confidence,
                                'method': 'tesseract',
                                'timestamp': datetime.now().isoformat()
                            }
            except Exception as e:
                continue
        
        return best_result
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        if image is None or image.size == 0:
            return image
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Resize if too small
        h, w = enhanced.shape
        if h < 30 or w < 60:
            scale = max(60/h, 100/w, 2.0)
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply threshold for binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _clean_text(self, text):
        """Clean and normalize OCR text"""
        if not text:
            return ""
        
        # Remove non-alphanumeric characters and convert to uppercase
        text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        # Common OCR corrections for license plates
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
    
    def _is_valid_plate_text(self, text):
        """Check if text looks like a valid license plate"""
        if not text or len(text) < 3 or len(text) > 12:
            return False
        
        # Must be alphanumeric
        if not text.isalnum():
            return False
        
        # Should have both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return (has_letter and has_number) or (text.isdigit() and len(text) >= 4)

# Global instance
multi_engine_ocr = MultiEngineOCR()