# Vehicle Identification System

A real-time AI-powered vehicle detection and license plate recognition system with web dashboard.

## Features

- **Real-time Vehicle Detection**: Uses YOLOv8 for detecting cars, motorcycles, buses, and trucks
- **License Plate Recognition**: Advanced OCR with multiple engines (EasyOCR, Tesseract)
- **Web Dashboard**: Real-time monitoring with statistics and live video feed
- **Vehicle Tracking**: Assigns unique IDs to vehicles and tracks their movement
- **Modular Architecture**: Clean, maintainable code structure

## Project Structure

```
Vehicle_Identification_/
├── config/                 # Configuration settings
├── core/                   # Core detection and tracking logic
├── database/              # Database management
├── license_plate/         # License plate detection modules
├── static/               # Web assets (CSS, JS)
├── templates/            # HTML templates
├── tests/                # Unit tests
├── utils/                # Utility modules
├── web/                  # Web dashboard
├── app.py               # Main application entry point
├── quick_test.py        # Quick testing script
└── requirements.txt     # Python dependencies
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Vehicle_Identification_
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO models** (if not present)
   - YOLOv8n model will be downloaded automatically
   - For license plate detection, ensure `yolov8_license_plate2 (1).pt` is in the root directory

## Usage

### Web Dashboard
```bash
python app.py
```
Access the dashboard at `http://localhost:9002`

### Quick Test
```bash
python quick_test.py
```

### Running Tests
```bash
python -m pytest tests/
```

## Configuration

Edit `config/settings.py` to customize:
- Camera settings
- Model parameters
- UI colors
- Detection thresholds

## Environment Variables

- `SECRET_KEY`: Flask secret key for production

## Features in Detail

### Vehicle Detection
- Supports multiple vehicle types: cars, motorcycles, buses, trucks
- Real-time tracking with unique vehicle IDs
- Confidence threshold filtering

### License Plate Recognition
- Multiple OCR engines for better accuracy
- Advanced image preprocessing
- Pattern validation for different regions

### Web Dashboard
- Live video feed with overlays
- Real-time statistics
- Vehicle entry/exit logging
- Export functionality

## Performance Optimization

- Uses deque for efficient data structures
- Caches recent license plates per vehicle
- Modular processing pipeline
- Configurable frame processing limits

## Troubleshooting

1. **No camera detected**: Check camera permissions and connections
2. **YOLO model not found**: Ensure model files are in the correct directory
3. **OCR not working**: Install Tesseract system package if using pytesseract

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use proper logging instead of print statements

## License

[Add your license information here]