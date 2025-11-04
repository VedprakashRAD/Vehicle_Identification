# 🚗 Unified Vehicle Identification System

A comprehensive vehicle identification system with real-time detection capabilities for cars, motorcycles, buses, and trucks.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

The system can be run in three different modes:

#### Web Dashboard Mode (Default)
```bash
python main.py --mode web
```
Access the dashboard at: `http://localhost:9090`

#### Mobile Mode
```bash
python main.py --mode mobile
```

#### Standalone Mode
```bash
python main.py --mode standalone
```

### 3. Advanced Usage
```bash
# Web Dashboard with custom host/port
python main.py --mode web --host 127.0.0.1 --port 8080

# Mobile Mode with specific camera
python main.py --mode mobile --camera 1

# Standalone Mode with video file
python main.py --mode standalone --source test_video.mp4
```

## 🎯 Features

- **Real-time Vehicle Detection**: Detects cars, motorcycles, buses, and trucks
- **License Plate Recognition**: Extracts license plate information
- **Employee Vehicle Management**: Tracks employee vehicles separately
- **Multi-mode Operation**: Web dashboard, mobile app, or standalone
- **Database Integration**: Stores detection history and employee data
- **Volkswagen Group Brand Recognition**: Specifically recognizes Skoda, Audi, Porsche, Lamborghini, and Bentley
- **Vehicle Entry/Exit Tracking**: Advanced system that pairs front and rear license plates
- **Anomaly Detection**: Identifies inconsistencies and flags for review
- **Time-Window Based Pairing**: Configurable time windows for matching plate detections

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- EasyOCR
- Flask (for web dashboard)

## 📱 Controls

- Press 'q' to quit in mobile and standalone modes
- Use web interface controls for the dashboard mode

## 🚗 Vehicle Entry/Exit Logic Implementation

This implementation provides a complete vehicle entry/exit tracking system that pairs front and rear license plates using time-window and vehicle attributes for accuracy.

### ✅ Vehicle Entry Logic
- Camera 1 (front-facing at entry) detects vehicle entering and captures front plate
- Camera 2 (rear-facing) detects and captures rear plate
- System pairs both plate numbers using time-window and vehicle attributes
- Entry event saved to database with complete information

### ✅ Vehicle Exit Logic
- Camera 2 (rear-facing) detects rear plate as vehicle leaves
- Camera 1 (entry camera) captures front plate as vehicle exits
- System pairs both plate numbers and completes the journey
- Exit event logged and matched with entry record

### ✅ Additional Capabilities
- Employee vehicles automatically flagged and categorized
- Anomaly detection for inconsistencies and missing data
- Real-time processing and event tracking
- Multi-vehicle support with simultaneous tracking

## Implementation Details

### Core Components

1. **VehiclePairingManager** (`core/vehicle_pairing.py`)
   - Manages pairing of front/rear license plates
   - Implements time-window based detection grouping
   - Handles entry/exit event creation and completion

2. **WorkingVehicleTracker** (`core/working_tracker.py`)
   - Processes video feeds from both cameras
   - Integrates with Ultimate ALPR Enhanced for plate detection
   - Sends detections to pairing system

3. **Database Storage**
   - SQLite database for persistent event storage
   - Temporary detection table for pairing
   - Permanent events table for completed journeys

### How It Works

1. **Entry Process**
   ```
   Camera 1 detects front plate → Camera 2 detects rear plate → Entry event created
   ```

2. **Exit Process**
   ```
   Camera 2 detects rear plate → Camera 1 detects front plate → Entry event completed
   ```

3. **Pairing Logic**
   - Detections grouped by time windows (30-second default)
   - Vehicle attributes (color, brand) matched for accuracy
   - Events created for complete plate pairs
   - Employee vehicles automatically flagged

## API Endpoints

### Vehicle Events
- `GET /api/paired_events` - Get paired vehicle events
- `GET /api/anomalies` - Get detected anomalies

### Employee Management
- `GET /api/employees` - List employee vehicles
- `POST /api/employees` - Add employee vehicle
- `DELETE /api/employees/<plate>` - Remove employee vehicle

## Testing

### Quick Test
```python
from core.vehicle_pairing import VehiclePairingManager
pm = VehiclePairingManager()

# Add entry
pm.add_plate_detection('ABC123', 1, 'red', 'Toyota', False)  # Camera 1
pm.add_plate_detection('ABC123', 2, 'red', 'Toyota', False)  # Camera 2

# Add exit
pm.add_plate_detection('ABC123', 2, 'red', 'Toyota', False)  # Camera 2
pm.add_plate_detection('ABC123', 1, 'red', 'Toyota', False)  # Camera 1

# Check completion
events = pm.get_recent_events()
completed = [e for e in events if e['status'] == 'completed']
```

## Configuration

### Pairing Window
Default: 30 seconds (configurable in VehiclePairingManager)

### Camera Setup
- Camera 1: Entry camera (front-facing at entry point)
- Camera 2: Exit camera (rear-facing at exit point)

## Database Schema

### Vehicle Events
```sql
CREATE TABLE vehicle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    front_plate TEXT,
    rear_plate TEXT,
    entry_time DATETIME,
    exit_time DATETIME,
    entry_camera_id INTEGER,
    exit_camera_id INTEGER,
    vehicle_color TEXT,
    vehicle_brand TEXT,
    is_employee BOOLEAN,
    status TEXT,  -- 'entry_pending', 'completed', 'anomaly'
    visit_count INTEGER DEFAULT 1
);
```

## Requirements Satisfied

✅ Cameras operate 24/7, recording video continuously
✅ Camera 1 detects front plate on entry
✅ Camera 2 detects rear plate on entry
✅ System pairs plates using time-window and attributes
✅ Entry event saved with complete data
✅ Camera 2 detects rear plate on exit
✅ Camera 1 detects front plate on exit
✅ System pairs plates for exit and completes journey
✅ Employee vehicles flagged automatically
✅ Anomalies detected and flagged for review

## 📄 License

This project is licensed under the MIT License.