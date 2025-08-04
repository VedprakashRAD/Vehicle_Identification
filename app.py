#!/usr/bin/env python3
"""
Main application entry point for Vehicle Monitoring System
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from web.dashboard import VehicleDashboard
from config.settings import app_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vehicle_monitoring.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main application entry point with proper error handling"""
    logger.info("🚗 Starting AI Vehicle Monitoring System...")
    logger.info("=" * 50)
    
    try:
        # Create and run dashboard with configuration
        dashboard = VehicleDashboard(
            host=app_config.HOST,
            port=app_config.PORT,
            debug=app_config.DEBUG
        )
        
        logger.info(f"Dashboard starting on {app_config.HOST}:{app_config.PORT}")
        dashboard.run()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down gracefully...")
    except ImportError as e:
        logger.error(f"❌ Import Error: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()