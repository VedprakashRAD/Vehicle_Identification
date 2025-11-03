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

## 📄 License

This project is licensed under the MIT License.
