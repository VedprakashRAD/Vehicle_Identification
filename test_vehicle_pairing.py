#!/usr/bin/env python3
"""
Test script for the Vehicle Pairing system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from core.vehicle_pairing import VehiclePairingManager
from datetime import datetime, timedelta

def test_vehicle_pairing():
    """Test the vehicle pairing system"""
    print("Testing Vehicle Pairing System...")
    
    # Initialize the pairing manager
    pairing_manager = VehiclePairingManager()
    
    # Test 1: Add individual plate detections
    print("\n1. Adding individual plate detections...")
    
    # Add front plate detection (Camera 1)
    success = pairing_manager.add_plate_detection(
        plate_number="MH12AB1234",
        camera_id=1,
        vehicle_color="red",
        vehicle_brand="Toyota",
        is_employee=False
    )
    print(f"   Front plate detection added: {'✅' if success else '❌'}")
    
    # Add rear plate detection (Camera 2)
    success = pairing_manager.add_plate_detection(
        plate_number="MH12AB1234",  # Same plate (for testing)
        camera_id=2,
        vehicle_color="red",
        vehicle_brand="Toyota",
        is_employee=False
    )
    print(f"   Rear plate detection added: {'✅' if success else '❌'}")
    
    # Test 2: Check recent events
    print("\n2. Checking recent events...")
    events = pairing_manager.get_recent_events(10)
    print(f"   Found {len(events)} events")
    
    for event in events:
        print(f"   Event ID: {event['id']}")
        print(f"   Front Plate: {event['front_plate']}")
        print(f"   Rear Plate: {event['rear_plate']}")
        print(f"   Entry Time: {event['entry_time']}")
        print(f"   Exit Time: {event['exit_time']}")
        print(f"   Status: {event['status']}")
        print()
    
    # Test 3: Check for anomalies
    print("\n3. Checking for anomalies...")
    anomalies = pairing_manager.check_for_anomalies()
    print(f"   Found {len(anomalies)} anomalies")
    
    for anomaly in anomalies:
        print(f"   Anomaly Type: {anomaly['type']}")
        print(f"   Message: {anomaly['message']}")
        print()
    
    # Test 4: Add employee vehicle detection
    print("\n4. Adding employee vehicle detection...")
    success = pairing_manager.add_plate_detection(
        plate_number="DL01EF5678",
        camera_id=1,
        vehicle_color="blue",
        vehicle_brand="Volkswagen",
        is_employee=True
    )
    print(f"   Employee vehicle detection added: {'✅' if success else '❌'}")
    
    # Add corresponding rear plate
    success = pairing_manager.add_plate_detection(
        plate_number="DL01EF5678",
        camera_id=2,
        vehicle_color="blue",
        vehicle_brand="Volkswagen",
        is_employee=True
    )
    print(f"   Employee rear plate detection added: {'✅' if success else '❌'}")
    
    # Check events again
    print("\n5. Checking events after employee vehicle...")
    events = pairing_manager.get_recent_events(10)
    employee_events = [e for e in events if e['is_employee']]
    print(f"   Found {len(employee_events)} employee vehicle events")
    
    for event in employee_events:
        print(f"   Employee Event - Front: {event['front_plate']}, Rear: {event['rear_plate']}")
    
    print("\n🎉 Vehicle Pairing System test completed!")

if __name__ == "__main__":
    test_vehicle_pairing()