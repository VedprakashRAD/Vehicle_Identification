import torch
import cv2
import numpy as np
from datetime import datetime
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

class WorkingVehicleTracker:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.total_count = 0
        self.active_tracks = 0
        self.entry_exit_log = []
        self.latest_stats = {}
        
    def process_frame_for_web(self, frame):
        # Simulate vehicle detection for demo
        processed_frame = frame.copy()
        
        # Add demo detection boxes
        cv2.rectangle(processed_frame, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.putText(processed_frame, 'Car', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update demo stats
        self.total_count += 1
        self.vehicle_counts['car'] += 1
        self.active_tracks = 1
        
        stats = {
            'total_count': self.total_count,
            'vehicle_counts': self.vehicle_counts.copy(),
            'active_tracks': self.active_tracks,
            'timestamp': datetime.now().isoformat(),
            'entry_exit_log': self.entry_exit_log
        }
        
        self.latest_stats = stats
        return processed_frame, stats
    
    def reset_counts(self):
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.total_count = 0
        self.active_tracks = 0
        self.entry_exit_log = []
        
    def get_vehicle_details(self):
        return [
            {
                'vehicle_id': 'V001',
                'registration_number': 'ABC-123',
                'vehicle_type': 'Car',
                'status': 'Entry',
                'entry_time': datetime.now().strftime('%H:%M:%S'),
                'exit_time': None
            }
        ]