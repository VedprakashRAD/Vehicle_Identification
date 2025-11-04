import sqlite3
import json
from datetime import datetime, timedelta

class DatabaseManager:
    def __init__(self, db_path='vehicle_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main vehicle logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                vehicle_type TEXT,
                count INTEGER
            )
        ''')
        
        # Employee vehicles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employee_vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE,
                employee_name TEXT,
                brand TEXT,
                department TEXT,
                registered_date DATETIME
            )
        ''')
        
        # Vehicle entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                vehicle_color TEXT,
                vehicle_brand TEXT,
                estimated_country TEXT DEFAULT 'unknown',
                is_enhanced BOOLEAN DEFAULT FALSE,
                entry_time DATETIME,
                exit_time DATETIME,
                is_employee BOOLEAN,
                camera_entry BOOLEAN,  -- TRUE for entry camera, FALSE for exit
                visit_count INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_employee_vehicle(self, plate_number, employee_name, brand, department):
        """Add an employee vehicle to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO employee_vehicles 
                (plate_number, employee_name, brand, department, registered_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (plate_number, employee_name, brand, department, datetime.now()))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding employee vehicle: {e}")
            return False
        finally:
            conn.close()
    
    def is_employee_vehicle(self, plate_number):
        """Check if a vehicle belongs to an employee"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM employee_vehicles 
            WHERE plate_number = ?
        ''', (plate_number,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] > 0 if result else False
    
    def log_vehicle_entry(self, plate_number, vehicle_color, vehicle_brand, is_employee, camera_entry, 
                         estimated_country='unknown', is_enhanced=False):
        """Log a vehicle entry with enhanced information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if this is a repeat visit by looking for previous entries without exit
            cursor.execute('''
                SELECT visit_count FROM vehicle_entries 
                WHERE plate_number = ? AND exit_time IS NULL
                ORDER BY entry_time DESC 
                LIMIT 1
            ''', (plate_number,))
            
            result = cursor.fetchone()
            visit_count = 1
            if result:
                # Vehicle is already in (no exit recorded), update visit count
                visit_count = result[0] + 1
                # Update the existing entry with new visit count
                cursor.execute('''
                    UPDATE vehicle_entries 
                    SET visit_count = ?, entry_time = ?
                    WHERE plate_number = ? AND exit_time IS NULL
                ''', (visit_count, datetime.now(), plate_number))
            else:
                # New entry or previous entry has exit time
                # Check for previous visits
                cursor.execute('''
                    SELECT MAX(visit_count) FROM vehicle_entries 
                    WHERE plate_number = ?
                ''', (plate_number,))
                
                result = cursor.fetchone()
                if result and result[0]:
                    visit_count = result[0] + 1
                
                # Insert new entry with enhanced information
                cursor.execute('''
                    INSERT INTO vehicle_entries 
                    (plate_number, vehicle_color, vehicle_brand, estimated_country, is_enhanced, 
                     entry_time, is_employee, camera_entry, visit_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (plate_number, vehicle_color, vehicle_brand, estimated_country, is_enhanced,
                      datetime.now(), is_employee, camera_entry, visit_count))
            
            conn.commit()
            print(f"🚗 Vehicle ENTRY logged: {plate_number} (Visit #{visit_count})")
        except Exception as e:
            print(f"Error logging vehicle entry: {e}")
        finally:
            conn.close()
    
    def log_vehicle_exit(self, plate_number):
        """Log a vehicle exit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Find the most recent entry without an exit time
            cursor.execute('''
                SELECT id FROM vehicle_entries 
                WHERE plate_number = ? AND exit_time IS NULL
                ORDER BY entry_time DESC 
                LIMIT 1
            ''', (plate_number,))
            
            result = cursor.fetchone()
            if result:
                # Update the exit time for this entry
                cursor.execute('''
                    UPDATE vehicle_entries 
                    SET exit_time = ?
                    WHERE id = ?
                ''', (datetime.now(), result[0]))
                
                conn.commit()
                print(f"🚗 Vehicle EXIT logged: {plate_number}")
            else:
                # No entry found, this might be an error or vehicle was already logged as exited
                print(f"⚠️ No active entry found for vehicle: {plate_number}")
        except Exception as e:
            print(f"Error logging vehicle exit: {e}")
        finally:
            conn.close()
    
    def get_recent_entries(self, limit=20):
        """Get recent vehicle entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT plate_number, vehicle_color, vehicle_brand, estimated_country, is_enhanced, entry_time, exit_time, is_employee, visit_count
            FROM vehicle_entries
            ORDER BY entry_time DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        entries = []
        for row in results:
            entries.append({
                'plate_number': row[0],
                'vehicle_color': row[1],
                'vehicle_brand': row[2],
                'estimated_country': row[3],
                'is_enhanced': row[4],
                'entry_time': row[5],
                'exit_time': row[6],
                'is_employee': row[7],
                'visit_count': row[8]
            })
        
        return entries
    
    def get_vehicle_history(self, plate_number):
        """Get history for a specific vehicle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT entry_time, exit_time, is_employee, visit_count
            FROM vehicle_entries
            WHERE plate_number = ?
            ORDER BY entry_time DESC
        ''', (plate_number,))
        
        results = cursor.fetchall()
        conn.close()
        
        history = []
        for row in results:
            history.append({
                'entry_time': row[0],
                'exit_time': row[1],
                'is_employee': row[2],
                'visit_count': row[3]
            })
        
        return history
    
    def get_hourly_summary(self, days=7):
        # Return demo data
        return [
            {'hour': (datetime.now() - timedelta(hours=i)).isoformat(), 'total': 10 + i}
            for i in range(24)
        ]
    
    def get_trend_data(self):
        return {'trend': 'increasing', 'percentage': 15.5}
    
    def get_model_insights(self):
        return {'accuracy': 95.2, 'confidence': 0.85}
    
    def export_data(self, format_type='csv'):
        if format_type == 'csv':
            return "timestamp,vehicle_type,count\n2024-01-01 10:00:00,car,5\n"
        return None