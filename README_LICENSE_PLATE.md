# AI Vehicle Monitoring System with License Plate Recognition

## 🚗 Features Implemented

### Vehicle Detection
- **YOLOv8 Integration**: Real-time detection of cars, motorcycles, buses, and trucks
- **Live Camera Support**: Works with webcam or IP cameras
- **Video File Support**: Can process pre-recorded videos
- **Real-time Statistics**: Live counting and tracking

### License Plate Recognition
- **Automatic Detection**: Detects license plates within detected vehicles
- **OCR Recognition**: Uses EasyOCR and Tesseract for text extraction
- **Multi-region Support**: Supports US, EU, and Indian license plate formats
- **Confidence Scoring**: Provides accuracy scores for detections

### Web Dashboard
- **Real-time Video Feed**: Live camera stream with overlays
- **Statistics Panel**: Shows vehicle counts and license plate detections
- **Vehicle Details Table**: Displays detected vehicles with license plates
- **Export Functionality**: Download detection data
- **Theme Support**: Light/dark mode toggle

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract (Optional)
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Run the Application
```bash
python3 app.py
```

### 4. Access Web Interface
Open your browser and go to: `http://localhost:9002`

## 📋 Usage Instructions

### Starting Detection
1. Open the web interface
2. Select video source (camera or test video)
3. Adjust confidence threshold if needed
4. Click "Start Monitoring"

### Features Available
- **Live Video Feed**: See real-time detection with bounding boxes
- **Vehicle Statistics**: Track counts by vehicle type
- **License Plate Recognition**: Automatic plate detection and text recognition
- **Data Export**: Download detection results as CSV
- **Reset Counts**: Clear statistics when needed

## 🎯 Detection Capabilities

### Vehicle Types Detected
- **Cars**: Green bounding boxes
- **Motorcycles**: Blue bounding boxes  
- **Buses**: Yellow bounding boxes
- **Trucks**: Magenta bounding boxes

### License Plate Recognition
- **Detection**: Yellow bounding boxes around license plates
- **Text Recognition**: OCR-extracted text displayed
- **Confidence Scores**: Accuracy percentage for each detection
- **Pattern Validation**: Validates against common license plate formats

## 📊 Data Output

### Real-time Statistics
- Total vehicle count
- Count by vehicle type
- Active detections
- License plate detections
- Timestamp information

### Vehicle Details Table
- Vehicle ID
- License plate number
- Vehicle type
- Entry/exit status
- Detection time
- Confidence score

## 🔧 Configuration Options

### Video Sources
- Default camera (index 0)
- Secondary camera (index 1)
- Vehicle test video (synthetic)
- Custom video files

### Detection Parameters
- Confidence threshold (0.1 - 1.0)
- OCR confidence threshold
- License plate validation patterns

## 🧪 Testing

### Test Scripts Available
- `test_license_plate.py`: Test license plate recognition
- `test_real_detection.py`: Test vehicle detection
- `quick_test.py`: Quick functionality test
- `test_camera.py`: Camera availability test

### Test Videos
- `vehicle_test_video.mp4`: Multi-vehicle test video
- `test_video.mp4`: Basic test video

## 📁 Project Structure

```
Vehicle_Identification_/
├── core/
│   └── working_tracker.py      # Main vehicle tracking logic
├── license_plate/
│   ├── __init__.py
│   └── detector.py             # License plate detection & OCR
├── web/
│   └── dashboard.py            # Flask web application
├── templates/
│   └── index.html              # Web interface
├── static/
│   └── css/                    # Stylesheets
├── weights/
│   └── best.pt                 # Custom model weights
├── models/                     # YOLOv5 model files
├── utils/                      # Utility modules
├── database/                   # Database management
├── app.py                      # Main application entry
└── requirements.txt            # Dependencies
```

## 🚀 Advanced Features

### License Plate Processing Pipeline
1. **Vehicle Detection**: YOLO detects vehicles in frame
2. **Region Extraction**: Crops vehicle region for plate search
3. **Plate Detection**: Finds license plate candidates
4. **Image Preprocessing**: Enhances image for better OCR
5. **Text Recognition**: Extracts text using OCR
6. **Validation**: Validates against known patterns
7. **Display**: Shows results with confidence scores

### OCR Engines
- **EasyOCR**: Primary OCR engine (GPU/CPU support)
- **Tesseract**: Fallback OCR engine
- **Preprocessing**: Image enhancement for better accuracy

### Pattern Matching
- **US Formats**: ABC1234, AB12345, 123ABC
- **EU Formats**: AB12CDE, A123BCD
- **Indian Formats**: MH12AB1234, MH12A1234

## 🔍 Troubleshooting

### Common Issues
1. **No camera detected**: Check camera permissions and connections
2. **Poor OCR accuracy**: Ensure good lighting and image quality
3. **No license plates detected**: Adjust confidence thresholds
4. **Performance issues**: Consider using GPU acceleration

### Performance Tips
- Use GPU for YOLO inference if available
- Adjust confidence thresholds based on use case
- Process every nth frame for better performance
- Use appropriate video resolution

## 📈 Future Enhancements

### Planned Features
- **Vehicle Tracking**: Track vehicles across frames
- **Speed Detection**: Calculate vehicle speeds
- **Database Integration**: Store detection history
- **Alert System**: Notifications for specific events
- **API Endpoints**: REST API for external integration
- **Mobile App**: Mobile interface for monitoring

### Model Improvements
- **Custom License Plate Model**: Train specific plate detection model
- **Regional Adaptations**: Support for more license plate formats
- **Night Vision**: Enhanced detection in low light
- **Multi-camera Support**: Process multiple camera feeds

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Run test scripts to isolate problems
3. Check system requirements and dependencies
4. Verify camera and video file access

## 🎉 Success Indicators

The system is working correctly when you see:
- ✅ Model loaded successfully
- ✅ License plate detector loaded successfully
- 🎯 License plate detected: [PLATE_TEXT] (confidence: X.XX)
- Real-time video feed with detection overlays
- Vehicle statistics updating in web interface
- License plate text appearing in vehicle details table