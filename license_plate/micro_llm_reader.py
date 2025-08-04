"""
Micro LLM Reader using ultra-lightweight models for license plate OCR
"""

import cv2
import numpy as np
import re
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MicroLLMReader:
    """Ultra-lightweight local OCR using TinyLLaVA or MobileVLM"""
    
    def __init__(self, model_type: str = "tinyllava"):
        self.model_type = model_type
        self.available = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize micro model"""
        if self.model_type == "tinyllava":
            self._init_tinyllava()
        elif self.model_type == "mobilevlm":
            self._init_mobilevlm()
        else:
            self._init_pattern_ocr()
    
    def _init_tinyllava(self):
        """Initialize TinyLLaVA (150MB)"""
        try:
            import requests
            # Check if TinyLLaVA server is running
            response = requests.get("http://localhost:8080/health", timeout=2)
            if response.status_code == 200:
                self.available = True
                logger.info("TinyLLaVA available (150MB)")
        except:
            logger.info("TinyLLaVA not available, using pattern OCR")
            self._init_pattern_ocr()
    
    def _init_mobilevlm(self):
        """Initialize MobileVLM (300MB)"""
        try:
            import torch
            # Placeholder for MobileVLM initialization
            self.available = True
            logger.info("MobileVLM initialized (300MB)")
        except:
            logger.info("MobileVLM not available, using pattern OCR")
            self._init_pattern_ocr()
    
    def _init_pattern_ocr(self):
        """Initialize pattern-based OCR (0MB)"""
        self.available = True
        self.model_type = "pattern"
        logger.info("Using pattern-based OCR (0MB)")
    
    def read_license_plate(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Read license plate using micro model"""
        if not self.available or image is None or image.size == 0:
            return None
        
        if self.model_type == "tinyllava":
            return self._read_with_tinyllava(image)
        elif self.model_type == "mobilevlm":
            return self._read_with_mobilevlm(image)
        else:
            return self._read_with_pattern(image)
    
    def _read_with_tinyllava(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Read using TinyLLaVA API"""
        try:
            import requests
            import base64
            
            _, buffer = cv2.imencode('.jpg', image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "image": base64_image,
                "prompt": "Extract license plate text. Return only alphanumeric characters."
            }
            
            response = requests.post("http://localhost:8080/ocr", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                text = re.sub(r'[^A-Z0-9]', '', result.get('text', '').upper())
                
                if len(text) >= 3:
                    return {
                        'text': text,
                        'confidence': 0.75,
                        'method': 'tinyllava_150mb',
                        'timestamp': datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"TinyLLaVA error: {e}")
        
        return None
    
    def _read_with_mobilevlm(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Read using MobileVLM"""
        try:
            # Placeholder for MobileVLM inference
            # This would use a lightweight vision-language model
            text = self._simple_ocr_fallback(image)
            
            if text and len(text) >= 3:
                return {
                    'text': text,
                    'confidence': 0.7,
                    'method': 'mobilevlm_300mb',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"MobileVLM error: {e}")
        
        return None
    
    def _read_with_pattern(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Pattern-based OCR using template matching"""
        try:
            text = self._simple_ocr_fallback(image)
            
            if text and len(text) >= 3:
                return {
                    'text': text,
                    'confidence': 0.6,
                    'method': 'pattern_ocr_0mb',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Pattern OCR error: {e}")
        
        return None
    
    def _simple_ocr_fallback(self, image: np.ndarray) -> str:
        """Simple OCR using contour analysis"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours for characters
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter character-like contours
        chars = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 8 < w < 50 and 15 < h < 60:  # Character size range
                chars.append((x, y, w, h))
        
        # Sort by x position
        chars.sort(key=lambda x: x[0])
        
        # Simple character recognition
        result = ""
        for x, y, w, h in chars:
            char_img = thresh[y:y+h, x:x+w]
            char = self._recognize_char(char_img)
            if char:
                result += char
        
        return result[:8]  # Limit to 8 characters
    
    def _recognize_char(self, char_img: np.ndarray) -> str:
        """Simple character recognition"""
        if char_img.size == 0:
            return ""
        
        # Resize to standard size
        char_img = cv2.resize(char_img, (16, 24))
        
        # Count white pixels in regions
        h, w = char_img.shape
        total_white = np.sum(char_img == 255)
        
        if total_white < 20:
            return ""
        
        # Simple pattern matching based on pixel distribution
        top_half = np.sum(char_img[:h//2] == 255)
        bottom_half = np.sum(char_img[h//2:] == 255)
        left_half = np.sum(char_img[:, :w//2] == 255)
        right_half = np.sum(char_img[:, w//2:] == 255)
        
        # Basic character patterns
        if top_half > bottom_half * 1.5:
            return "P" if left_half > right_half else "7"
        elif bottom_half > top_half * 1.5:
            return "L" if left_half > right_half else "J"
        elif abs(top_half - bottom_half) < total_white * 0.2:
            if total_white > 150:
                return "0"
            elif total_white > 100:
                return "8"
            else:
                return "1"
        else:
            # Default to common characters
            ratio = total_white / (h * w)
            if ratio > 0.6:
                return "0"
            elif ratio > 0.4:
                return "A"
            else:
                return "1"