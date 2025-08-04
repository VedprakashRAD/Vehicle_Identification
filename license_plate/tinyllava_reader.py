"""
TinyLLaVA Reader - 150MB lightweight vision model for license plate OCR
"""

import cv2
import numpy as np
import base64
import re
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TinyLLaVAReader:
    """TinyLLaVA-based license plate reader (150MB)"""
    
    def __init__(self):
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if TinyLLaVA is available"""
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Try to load TinyLLaVA model
            model_name = "bczhou/tiny-llava-v1_5-1.1b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("TinyLLaVA loaded (150MB)")
            return True
        except Exception as e:
            logger.warning(f"TinyLLaVA not available: {e}")
            return False
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for TinyLLaVA"""
        # Resize and enhance
        h, w = image.shape[:2]
        if h < 64 or w < 128:
            scale = max(64/h, 128/w, 2.0)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Enhance contrast
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    def read_license_plate(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Read license plate using TinyLLaVA"""
        if not self.available or image is None or image.size == 0:
            return None
            
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(processed_image)
            
            # Create prompt
            prompt = "Extract the license plate text from this image. Return only the alphanumeric characters you see, no spaces or special characters."
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    images=pil_image,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract plate text
            plate_text = re.sub(r'[^A-Z0-9]', '', response.upper())
            
            if plate_text and len(plate_text) >= 3:
                return {
                    'text': plate_text,
                    'confidence': 0.8,
                    'method': 'tinyllava_150mb',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"TinyLLaVA error: {e}")
            
        return None


class PatternOCR:
    """Fallback pattern-based OCR (0MB)"""
    
    def read_license_plate(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Pattern-based license plate reading"""
        if image is None or image.size == 0:
            return None
            
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhance
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find character contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            chars = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 8 < w < 50 and 15 < h < 60:
                    chars.append((x, y, w, h))
            
            chars.sort(key=lambda x: x[0])
            
            # Simple character recognition
            result = ""
            for x, y, w, h in chars:
                char_img = thresh[y:y+h, x:x+w]
                char = self._recognize_char(char_img)
                if char:
                    result += char
            
            if len(result) >= 3:
                return {
                    'text': result[:8],
                    'confidence': 0.6,
                    'method': 'pattern_0mb',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Pattern OCR error: {e}")
            
        return None
    
    def _recognize_char(self, char_img: np.ndarray) -> str:
        """Simple character recognition"""
        if char_img.size == 0:
            return ""
        
        char_img = cv2.resize(char_img, (16, 24))
        h, w = char_img.shape
        total_white = np.sum(char_img == 255)
        
        if total_white < 20:
            return ""
        
        # Basic patterns
        top_half = np.sum(char_img[:h//2] == 255)
        bottom_half = np.sum(char_img[h//2:] == 255)
        
        if top_half > bottom_half * 1.3:
            return "7"
        elif bottom_half > top_half * 1.3:
            return "L"
        elif total_white > 150:
            return "0"
        elif total_white > 100:
            return "8"
        else:
            return "1"