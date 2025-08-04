"""
Configuration settings for Vehicle Identification System
"""
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class AppConfig:
    """Application configuration"""
    HOST: str = '0.0.0.0'
    PORT: int = 9002
    DEBUG: bool = True
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    
@dataclass
class ModelConfig:
    """Model configuration"""
    CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_MODEL_PATH: str = 'yolov8n.pt'
    LICENSE_PLATE_MODEL_PATH: str = 'yolov8_license_plate2 (1).pt'
    
    # Vehicle class mapping (COCO dataset)
    VEHICLE_CLASSES: Dict[int, str] = None
    
    def __post_init__(self):
        if self.VEHICLE_CLASSES is None:
            self.VEHICLE_CLASSES = {
                2: 'car',
                3: 'motorcycle', 
                5: 'bus',
                7: 'truck'
            }

@dataclass
class CameraConfig:
    """Camera configuration"""
    DEFAULT_WIDTH: int = 640
    DEFAULT_HEIGHT: int = 480
    DEFAULT_FPS: int = 30
    CAMERA_INDICES: List[int] = None
    
    def __post_init__(self):
        if self.CAMERA_INDICES is None:
            self.CAMERA_INDICES = [0, 1, 2]

@dataclass
class LLMConfig:
    """Micro LLM configuration"""
    ENABLED: bool = True
    MODEL_TYPE: str = "pattern"  # "tinyllava" (150MB), "mobilevlm" (300MB), "pattern" (0MB)
    CONFIDENCE_THRESHOLD: float = 0.6

@dataclass
class UIConfig:
    """UI configuration"""
    COLORS: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.COLORS is None:
            self.COLORS = {
                'car': (0, 255, 0),      # Green
                'motorcycle': (255, 0, 0), # Red
                'bus': (0, 255, 255),    # Yellow
                'truck': (255, 0, 255)   # Magenta
            }

# Global configuration instances
app_config = AppConfig()
model_config = ModelConfig()
camera_config = CameraConfig()
llm_config = LLMConfig()
ui_config = UIConfig()