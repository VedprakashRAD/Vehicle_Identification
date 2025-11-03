#!/usr/bin/env python3
"""
Employee Vehicle Management System
"""

import json
from database.manager import DatabaseManager
from typing import List, Dict

class EmployeeManager:
    def __init__(self):
        self.db = DatabaseManager()
    
    def add_employee_vehicle(self, plate_number: str, employee_name: str, brand: str, department: str) -> bool:
        """
        Add an employee vehicle to the system
        
        Args:
            plate_number: Vehicle registration number
            employee_name: Name of the employee
            brand: Vehicle brand (Skoda, Audi, Porsche, etc.)
            department: Employee department
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.db.add_employee_vehicle(plate_number, employee_name, brand, department)
    
    def bulk_import_employees(self, employee_data: List[Dict]) -> int:
        """
        Bulk import employee vehicles from a list of dictionaries
        
        Args:
            employee_data: List of dictionaries with employee vehicle data
            
        Returns:
            int: Number of successfully imported records
        """
        success_count = 0
        for record in employee_data:
            if self.add_employee_vehicle(
                record['plate_number'],
                record['employee_name'],
                record['brand'],
                record['department']
            ):
                success_count += 1
        return success_count
    
    def is_employee_vehicle(self, plate_number: str) -> bool:
        """
        Check if a vehicle belongs to an employee
        
        Args:
            plate_number: Vehicle registration number
            
        Returns:
            bool: True if vehicle belongs to an employee, False otherwise
        """
        return self.db.is_employee_vehicle(plate_number)
    
    def get_employee_vehicles(self) -> List[Dict]:
        """
        Get all registered employee vehicles
        
        Returns:
            List[Dict]: List of employee vehicle records
        """
        # This would query the database in a real implementation
        # For now, returning sample data
        return [
            {
                'plate_number': 'KA01AB1234',
                'employee_name': 'John Doe',
                'brand': 'Skoda',
                'department': 'Service'
            },
            {
                'plate_number': 'KA02CD5678',
                'employee_name': 'Jane Smith',
                'brand': 'Audi',
                'department': 'Sales'
            }
        ]
    
    def remove_employee_vehicle(self, plate_number: str) -> bool:
        """
        Remove an employee vehicle from the system
        
        Args:
            plate_number: Vehicle registration number
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Implementation would remove from database
        return True

# Sample usage
if __name__ == "__main__":
    # Sample employee data for Volkswagen workshop
    sample_employees = [
        {
            'plate_number': 'KA01AB1234',
            'employee_name': 'Rajesh Kumar',
            'brand': 'Skoda',
            'department': 'Service'
        },
        {
            'plate_number': 'KA02CD5678',
            'employee_name': 'Priya Sharma',
            'brand': 'Audi',
            'department': 'Sales'
        },
        {
            'plate_number': 'KA03EF9012',
            'employee_name': 'Amit Patel',
            'brand': 'Porsche',
            'department': 'Management'
        }
    ]
    
    # Initialize manager
    emp_manager = EmployeeManager()
    
    # Bulk import sample data
    imported = emp_manager.bulk_import_employees(sample_employees)
    print(f"Successfully imported {imported} employee vehicles")
    
    # Check if a vehicle is an employee vehicle
    test_plate = 'KA01AB1234'
    if emp_manager.is_employee_vehicle(test_plate):
        print(f"Vehicle {test_plate} belongs to an employee")
    else:
        print(f"Vehicle {test_plate} does not belong to an employee")