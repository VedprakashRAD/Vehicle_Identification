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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                vehicle_type TEXT,
                count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
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