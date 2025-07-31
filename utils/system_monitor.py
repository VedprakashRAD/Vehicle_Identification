import psutil
import threading
import time

class SystemMonitor:
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'gpu_usage': 0
        }
    
    def start_monitoring(self):
        self.monitoring = True
        thread = threading.Thread(target=self._monitor_loop)
        thread.daemon = True
        thread.start()
    
    def _monitor_loop(self):
        while self.monitoring:
            self.metrics['cpu_usage'] = psutil.cpu_percent()
            self.metrics['memory_usage'] = psutil.virtual_memory().percent
            self.metrics['disk_usage'] = psutil.disk_usage('/').percent
            time.sleep(5)
    
    def get_metrics(self):
        return self.metrics

system_monitor = SystemMonitor()