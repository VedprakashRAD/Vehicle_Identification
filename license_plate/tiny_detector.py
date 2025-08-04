"""
Tiny Enhanced Detector using TinyLLaVA (150MB) with pattern fallback
"""

from .intelligent_detector import IntelligentLicensePlateDetector
from .tinyllava_reader import TinyLLaVAReader, PatternOCR
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class TinyEnhancedDetector(IntelligentLicensePlateDetector):
    """Enhanced detector using TinyLLaVA (150MB)"""
    
    def __init__(self):
        super().__init__()
        self.tiny_llava = TinyLLaVAReader()
        self.pattern_ocr = PatternOCR()
        
    def recognize_text_enhanced(self, plate_image) -> Optional[Dict[str, Any]]:
        """Enhanced recognition with TinyLLaVA + pattern fallback"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Try TinyLLaVA first
        if self.tiny_llava.available:
            result = self.tiny_llava.read_license_plate(plate_image)
            if result:
                return result
        
        # Try traditional OCR
        traditional_result = self.recognize_text(plate_image)
        if traditional_result and traditional_result.get('confidence', 0) > 0.7:
            return traditional_result
        
        # Fallback to pattern OCR
        pattern_result = self.pattern_ocr.read_license_plate(plate_image)
        if pattern_result:
            return pattern_result
        
        return traditional_result
    
    def process_vehicle_for_plates(self, image, vehicle_bbox, vehicle_id) -> List[Dict[str, Any]]:
        """Process with TinyLLaVA enhancement"""
        results = []
        
        try:
            plates = self.detect_license_plates(image, vehicle_bbox)
            
            for plate in plates:
                recognition_result = self.recognize_text_enhanced(plate['image'])
                
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
                    
                    logger.info(f"🎯 TinyLLaVA plate: {recognition_result['text']} "
                              f"(method: {recognition_result.get('method')})")
                    break
        
        except Exception as e:
            logger.error(f"❌ TinyLLaVA detection error: {e}")
        
        return results