#!/usr/bin/env python3
"""
Install and verify dependencies for license plate detection
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} is available")
        return True
    except ImportError:
        print(f"❌ {module_name} is not available")
        if package_name:
            print(f"   Try installing: pip install {package_name}")
        return False

def main():
    print("🔧 Checking and installing dependencies for license plate detection...")
    
    # Required packages
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("cv2", "opencv-python"),
        ("ultralytics", "ultralytics"),
        ("easyocr", "easyocr"),
        ("pytesseract", "pytesseract"),
        ("numpy", "numpy"),
        ("PIL", "Pillow")
    ]
    
    missing_packages = []
    
    # Check what's missing
    print("\n📋 Checking current installations...")
    for module, package in packages:
        if not check_import(module):
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            install_package(package)
    else:
        print("\n✅ All required packages are already installed!")
    
    # Special check for Tesseract OCR system dependency
    print("\n🔍 Checking Tesseract OCR system installation...")
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            print(f"   Version: {result.stdout.split()[1]}")
        else:
            print("❌ Tesseract OCR not found")
            print("   Install with: brew install tesseract (macOS)")
    except FileNotFoundError:
        print("❌ Tesseract OCR not found")
        print("   Install with: brew install tesseract (macOS)")
    
    # Test imports after installation
    print("\n🧪 Testing imports after installation...")
    all_good = True
    for module, package in packages:
        if not check_import(module):
            all_good = False
    
    if all_good:
        print("\n🎉 All dependencies are ready!")
        print("You can now run the license plate detection system.")
    else:
        print("\n⚠️ Some dependencies are still missing.")
        print("Please install them manually and try again.")
    
    # Test license plate detector specifically
    print("\n🔬 Testing license plate detector...")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from license_plate.detector import LicensePlateDetector
        detector = LicensePlateDetector()
        print("✅ License plate detector initialized successfully")
        
        # Check if models are loaded
        if detector.plate_model is not None:
            print("✅ YOLO model loaded")
        else:
            print("⚠️ YOLO model not loaded")
            
        if detector.ocr_reader is not None:
            print("✅ EasyOCR loaded")
        else:
            print("⚠️ EasyOCR not loaded, will try Tesseract")
            
    except Exception as e:
        print(f"❌ Error testing license plate detector: {e}")

if __name__ == "__main__":
    main()