#!/usr/bin/env python3
"""
Main application entry point for Vehicle Monitoring System
"""

import logging
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web.dashboard import VehicleDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main application entry point"""
    print("🚗 Starting AI Vehicle Monitoring System...")
    print("=" * 50)
    
    # Create and run dashboard
    dashboard = VehicleDashboard(
        host='0.0.0.0',
        port=9002,
        debug=True
    )
    
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()