# Mobile Vehicle Identification System

A lightweight, mobile-optimized version of the Vehicle Identification System for workshop vehicle tracking with real-time ANPR (Automatic Number Plate Recognition).

## Features

- **Real-time Vehicle Detection**: Uses YOLOv8 Nano model for efficient vehicle detection
- **License Plate Recognition**: EasyOCR for accurate Indian license plate reading
- **Dual Camera Support**: Entry/Exit tracking with same-gate dual camera setup
- **Employee Vehicle Management**: Database of employee vehicles with VW Group brands
- **Mobile-Optimized**: Lightweight processing for smartphone deployment
- **Offline Operation**: Works without internet connection after initial setup

## System Requirements

### Hardware
- Android smartphone with camera (Android 8.0+ recommended)
- Minimum 4GB RAM
- At least 2GB free storage space
- Dual camera setup recommended (front/back cameras)

### Software Dependencies
See [requirements_mobile.txt](requirements_mobile.txt) for detailed dependencies.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mobile_optimized
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_mobile.txt
   ```

3. **Download YOLOv8 models** (if not already present):
   - `yolov8n.pt` (vehicle detection model)
   - `yolov8_license_plate2 (1).pt` (license plate detection model)

## Usage

### Mobile App Mode
Run the mobile-optimized ANPR system:
```bash
python mobile_anpr.py
```

### Web Dashboard Mode
Start the web-based dashboard for monitoring:
```bash
python app.py
```

Access the dashboard at `http://localhost:9090`

## Key Components

### 1. Mobile ANPR System (`mobile_anpr.py`)
- Lightweight processing optimized for mobile devices
- Frame skipping for better performance
- CPU-based processing (no GPU required)
- SQLite database for local storage

### 2. Core Vehicle Tracking (`core/working_tracker.py`)
- YOLOv8-based vehicle detection
- Real-time processing with tracking
- License plate recognition with EasyOCR
- Entry/Exit logging with database storage

### 3. Web Dashboard (`web/dashboard.py`)
- Flask-based web interface
- Real-time video streaming
- Statistics and analytics
- Employee vehicle management

## Supported Vehicle Brands

- **Volkswagen Group**: Skoda, Audi, Porsche, Lamborghini, Bentley, Volkswagen
- **Other vehicles**: Cars, motorcycles, buses, trucks

## Database Structure

The system uses SQLite databases for local storage:
- `vehicle_data.db`: Main vehicle detection database
- `mobile_workshop.db`: Mobile-optimized database

## Performance Optimization

### Mobile Settings
- Frame processing every 3rd frame
- Reduced resolution processing (640x480 max)
- CPU-only processing for compatibility
- Lightweight models (YOLOv8 Nano)

### Camera Configuration
- Entry camera: Front camera (index 0)
- Exit camera: Rear camera (index 1)
- 15 FPS capture rate for smooth operation

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Ensure camera permissions are granted
   - Check camera indices in the code
   - Test with different camera sources

2. **Model loading errors**:
   - Verify model files are in the correct location
   - Check file permissions
   - Ensure sufficient storage space

3. **OCR accuracy issues**:
   - Ensure good lighting conditions
   - Clean license plates
   - Adjust camera angle for better plate visibility

### Performance Tips

1. **Battery Optimization**:
   - Use power-saving mode
   - Close other apps during operation
   - Consider external power source for extended use

2. **Processing Speed**:
   - Reduce confidence threshold if needed
   - Use better lighting conditions
   - Position camera at optimal distance (2-5 meters)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and feature requests, please create an issue in the repository.