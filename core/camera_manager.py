"""
Camera Management Module
Handles camera initialization and video source management
"""
import cv2
import logging
from typing import Optional, List
from config.settings import camera_config

logger = logging.getLogger(__name__)

class CameraManager:
    """Manages camera initialization and video sources"""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_source = None
        
    def initialize_camera(self, preferred_source: int = 0) -> bool:
        """
        Initialize camera with fallback options
        
        Args:
            preferred_source: Preferred camera index
            
        Returns:
            bool: True if camera initialized successfully
        """
        # Try preferred source first
        if self._try_camera(preferred_source):
            return True
            
        # Try other camera indices
        for camera_idx in camera_config.CAMERA_INDICES:
            if camera_idx != preferred_source and self._try_camera(camera_idx):
                return True
                
        # Try video files as fallback
        video_files = ["vehicle_test_video.mp4", "test_video.mp4"]
        for video_file in video_files:
            if self._try_video_file(video_file):
                return True
                
        logger.warning("No camera or video source available")
        return False
    
    def _try_camera(self, camera_idx: int) -> bool:
        """Try to initialize a specific camera index"""
        try:
            logger.info(f"Trying camera index {camera_idx}")
            cap = cv2.VideoCapture(camera_idx)
            
            if cap.isOpened():
                # Test if camera actually works
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    logger.info(f"Camera {camera_idx} initialized successfully")
                    
                    # Configure camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.DEFAULT_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.DEFAULT_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, camera_config.DEFAULT_FPS)
                    
                    self.cap = cap
                    self.current_source = camera_idx
                    return True
                else:
                    logger.warning(f"Camera {camera_idx} opened but can't read frames")
            else:
                logger.warning(f"Camera {camera_idx} failed to open")
                
            cap.release()
            return False
            
        except Exception as e:
            logger.error(f"Error trying camera {camera_idx}: {e}")
            return False
    
    def _try_video_file(self, video_file: str) -> bool:
        """Try to open a video file"""
        try:
            logger.info(f"Trying video file: {video_file}")
            cap = cv2.VideoCapture(video_file)
            
            if cap.isOpened():
                logger.info(f"Video file {video_file} opened successfully")
                self.cap = cap
                self.current_source = video_file
                return True
            else:
                logger.warning(f"Failed to open video file: {video_file}")
                
        except Exception as e:
            logger.error(f"Error opening video file {video_file}: {e}")
            
        return False
    
    def read_frame(self) -> tuple[bool, Optional[any]]:
        """Read a frame from the current source"""
        if self.cap is None:
            return False, None
            
        try:
            ret, frame = self.cap.read()
            
            # Handle video file looping
            if not ret and isinstance(self.current_source, str):
                # Reset video to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                
            return ret, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def is_available(self) -> bool:
        """Check if camera source is available"""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.current_source = None
            logger.info("Camera released")
    
    def get_source_info(self) -> dict:
        """Get information about current source"""
        return {
            'source': self.current_source,
            'is_available': self.is_available(),
            'type': 'camera' if isinstance(self.current_source, int) else 'video'
        }