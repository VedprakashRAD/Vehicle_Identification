"""
Unit tests for Vehicle Tracker
"""
import unittest
import numpy as np
from core.working_tracker import WorkingVehicleTracker

class TestVehicleTracker(unittest.TestCase):
    
    def setUp(self):
        self.tracker = WorkingVehicleTracker(confidence_threshold=0.5)
    
    def test_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(self.tracker.total_count, 0)
        self.assertEqual(self.tracker.active_tracks, 0)
        self.assertIsInstance(self.tracker.vehicle_counts, dict)
    
    def test_reset_counts(self):
        """Test resetting counters"""
        # Simulate some counts
        self.tracker.total_count = 10
        self.tracker.vehicle_counts['car'] = 5
        
        # Reset
        self.tracker.reset_counts()
        
        # Verify reset
        self.assertEqual(self.tracker.total_count, 0)
        self.assertEqual(self.tracker.vehicle_counts['car'], 0)
    
    def test_vehicle_tracking(self):
        """Test vehicle tracking functionality"""
        bbox = (100, 100, 200, 200)
        vehicle_id = self.tracker._track_vehicle(bbox, 'car')
        
        self.assertIsInstance(vehicle_id, str)
        self.assertTrue(vehicle_id.startswith('V'))
        self.assertIn(vehicle_id, self.tracker.tracked_vehicles)
    
    def test_demo_detection(self):
        """Test demo detection mode"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed_frame, stats = self.tracker._demo_detection(frame)
        
        self.assertIsInstance(processed_frame, np.ndarray)
        self.assertIsInstance(stats, dict)
        self.assertIn('total_count', stats)
        self.assertIn('vehicle_counts', stats)

if __name__ == '__main__':
    unittest.main()