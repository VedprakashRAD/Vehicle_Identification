#!/usr/bin/env python3
"""
Mobile Vehicle Identification System - Quick Start Script
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Mobile Vehicle Identification System')
    parser.add_argument('--mode', choices=['web', 'mobile'], default='mobile',
                       help='Run mode: web dashboard or mobile app')
    parser.add_argument('--port', type=int, default=9090,
                       help='Port for web dashboard (default: 9090)')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        print("🚀 Starting Web Dashboard...")
        print(f"🌐 Access dashboard at: http://localhost:{args.port}")
        os.system(f"python app.py")
    else:
        print("📱 Starting Mobile ANPR System...")
        print("📷 Press 'q' to quit")
        os.system("python mobile_anpr.py")

if __name__ == "__main__":
    main()