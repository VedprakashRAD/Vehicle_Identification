from datetime import datetime

class AnomalyDetector:
    def __init__(self):
        self.active_anomalies = []
        self.recent_anomalies = []
    
    def get_active_anomalies(self):
        return self.active_anomalies
    
    def get_recent_anomalies(self, limit=10):
        return self.recent_anomalies[:limit]
    
    def get_anomaly_summary(self):
        return {
            'total_anomalies': len(self.recent_anomalies),
            'active_count': len(self.active_anomalies),
            'last_detected': datetime.now().isoformat() if self.recent_anomalies else None
        }

anomaly_detector = AnomalyDetector()