"""
Web Dashboard Module
Flask-based web interface for the vehicle monitoring system.
"""

from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit
import cv2
import threading
import time
import numpy as np
import json
from datetime import datetime
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.working_tracker import WorkingVehicleTracker
from database.manager import DatabaseManager
from employee_manager import EmployeeManager

logger = logging.getLogger(__name__)


def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class VehicleDashboard:
    """Main dashboard application class"""
    
    def __init__(self, host='0.0.0.0', port=9090, debug=True, config=None):
        self.host = host
        self.port = port
        self.debug = debug
        self.config = config or {}
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='../templates',
                        static_folder='../static')
        self.app.config['SECRET_KEY'] = 'vehicle_monitoring_secret_key'
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize database and employee manager
        self.db = DatabaseManager()
        self.emp_manager = EmployeeManager()
        
        # Application state for dual cameras
        self.vehicle_counter_entry = None
        self.vehicle_counter_exit = None
        self.is_processing = False
        self.camera_sources = {'entry': 0, 'exit': 1}  # Default camera sources
        
        # Setup routes and events
        self._setup_routes()
        self._setup_socket_events()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/stats')
        def get_stats():
            # Combine stats from both cameras
            combined_stats = {
                'total_count': 0,
                'vehicle_counts': {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0},
                'active_tracks': 0,
                'timestamp': datetime.now().isoformat(),
                'entry_exit_log': [],
                'license_plates': []
            }
            
            # Get stats from entry camera if available
            if self.vehicle_counter_entry and hasattr(self.vehicle_counter_entry, 'latest_stats'):
                entry_stats = self.vehicle_counter_entry.latest_stats
                combined_stats['total_count'] += entry_stats.get('total_count', 0)
                combined_stats['active_tracks'] += entry_stats.get('active_tracks', 0)
                
                # Combine vehicle counts
                for vehicle_type in combined_stats['vehicle_counts']:
                    combined_stats['vehicle_counts'][vehicle_type] += entry_stats.get('vehicle_counts', {}).get(vehicle_type, 0)
                
                # Combine license plates
                if 'license_plates' in entry_stats:
                    combined_stats['license_plates'].extend(entry_stats['license_plates'])
            
            # Get stats from exit camera if available
            if self.vehicle_counter_exit and hasattr(self.vehicle_counter_exit, 'latest_stats'):
                exit_stats = self.vehicle_counter_exit.latest_stats
                combined_stats['total_count'] += exit_stats.get('total_count', 0)
                combined_stats['active_tracks'] += exit_stats.get('active_tracks', 0)
                
                # Combine vehicle counts
                for vehicle_type in combined_stats['vehicle_counts']:
                    combined_stats['vehicle_counts'][vehicle_type] += exit_stats.get('vehicle_counts', {}).get(vehicle_type, 0)
                
                # Combine license plates
                if 'license_plates' in exit_stats:
                    combined_stats['license_plates'].extend(exit_stats['license_plates'])
            
            return jsonify(combined_stats)
        
        @self.app.route('/api/hourly_summary')
        def get_hourly_summary():
            days = int(request.args.get('days', 7))
            data = self.db.get_hourly_summary(days)
            return jsonify(data)
        
        @self.app.route('/api/trend_data')
        def get_trend_data():
            data = self.db.get_trend_data()
            return jsonify(data)
        
        @self.app.route('/api/model_insights')
        def get_model_insights():
            data = self.db.get_model_insights()
            return jsonify(data)
        
        @self.app.route('/api/export_data')
        def export_data():
            format_type = request.args.get('format', 'csv').lower()
            data = self.db.export_data(format_type)
            
            if data is None:
                return jsonify({'error': 'Export failed'}), 500
            
            if format_type == 'csv':
                return Response(data, mimetype='text/csv',
                              headers={'Content-disposition': 'attachment; filename=vehicle_data.csv'})
            elif format_type == 'json':
                return Response(data, mimetype='application/json',
                              headers={'Content-disposition': 'attachment; filename=vehicle_data.json'})
            elif format_type == 'xml':
                return Response(data, mimetype='application/xml',
                              headers={'Content-disposition': 'attachment; filename=vehicle_data.xml'})
            else:
                return jsonify({'error': 'Unsupported format'}), 400
        
        @self.app.route('/api/recent_entries')
        def get_recent_entries():
            """Get recent vehicle entries"""
            try:
                limit = int(request.args.get('limit', 20))
                entries = self.db.get_recent_entries(limit)
                return jsonify({'status': 'success', 'data': entries})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/vehicle_history/<plate_number>')
        def get_vehicle_history(plate_number):
            """Get history for a specific vehicle"""
            try:
                history = self.db.get_vehicle_history(plate_number)
                return jsonify({'status': 'success', 'data': history})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/api/entry_exit_log')
        def get_entry_exit_log():
            # Combine logs from both cameras
            combined_log = []
            if self.vehicle_counter_entry:
                combined_log.extend(list(self.vehicle_counter_entry.entry_exit_log))
            if self.vehicle_counter_exit:
                combined_log.extend(list(self.vehicle_counter_exit.entry_exit_log))
            return jsonify(combined_log)
        
        @self.app.route('/api/vehicle_details')
        def get_vehicle_details():
            # Combine vehicle details from both cameras
            combined_details = []
            
            if self.vehicle_counter_entry:
                entry_details = self.vehicle_counter_entry.get_vehicle_details()
                combined_details.extend(entry_details)
                
            if self.vehicle_counter_exit:
                exit_details = self.vehicle_counter_exit.get_vehicle_details()
                combined_details.extend(exit_details)
            
            print(f"Vehicle details API returning: {len(combined_details)} entries")
            return jsonify(combined_details)

        @self.app.route('/start_monitoring', methods=['POST'])
        def start_monitoring():
            try:
                data = request.get_json()
                source = data.get('source', 0)
                confidence = float(data.get('confidence', 0.5))
                camera_type = data.get('camera_type', 'entry')  # 'entry' or 'exit'
                
                # Set camera source based on type
                self.camera_sources[camera_type] = source
                
                # Initialize appropriate vehicle counter
                if camera_type == 'entry':
                    self.vehicle_counter_entry = WorkingVehicleTracker(confidence_threshold=confidence)
                else:  # exit
                    self.vehicle_counter_exit = WorkingVehicleTracker(confidence_threshold=confidence)
                
                self.is_processing = True
                
                logger.info(f"Started monitoring {camera_type} camera with source: {source}, confidence: {confidence}")
                return jsonify({'status': 'success', 'message': f'{camera_type.capitalize()} camera monitoring started'})
                
            except Exception as e:
                logger.error(f"Error starting monitoring: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/stop_monitoring', methods=['POST'])
        def stop_monitoring():
            self.is_processing = False
            if self.vehicle_counter_entry:
                self.vehicle_counter_entry.reset_counts()
            if self.vehicle_counter_exit:
                self.vehicle_counter_exit.reset_counts()
            logger.info("Monitoring stopped and counts reset")
            return jsonify({'status': 'success', 'message': 'Monitoring stopped'})
        
        @self.app.route('/reset_counts', methods=['POST'])
        def reset_counts():
            if self.vehicle_counter_entry:
                self.vehicle_counter_entry.reset_counts()
            if self.vehicle_counter_exit:
                self.vehicle_counter_exit.reset_counts()
            return jsonify({'status': 'success', 'message': 'Counts reset successfully'})
        
        @self.app.route('/api/force_update')
        def force_update():
            # Force emit current stats from both cameras
            stats = self._get_combined_stats()
            self.socketio.emit('stats_update', stats)
            return jsonify({'status': 'success', 'stats': stats})
        
        @self.app.route('/video_feed')
        def video_feed():
            camera_id = request.args.get('camera', '1')
            return Response(self._generate_frames(camera_id),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        # Employee Management API Endpoints
        @self.app.route('/api/employees', methods=['GET'])
        def get_employees():
            """Get all employee vehicles"""
            try:
                employees = self.emp_manager.get_employee_vehicles()
                return jsonify({'status': 'success', 'data': employees})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/employees', methods=['POST'])
        def add_employee():
            """Add a new employee vehicle"""
            try:
                data = request.get_json()
                required_fields = ['plate_number', 'employee_name', 'brand', 'department']
                
                # Validate required fields
                for field in required_fields:
                    if field not in data:
                        return jsonify({'status': 'error', 'message': f'Missing required field: {field}'}), 400
                
                # Add employee vehicle
                success = self.emp_manager.add_employee_vehicle(
                    data['plate_number'],
                    data['employee_name'],
                    data['brand'],
                    data['department']
                )
                
                if success:
                    return jsonify({'status': 'success', 'message': 'Employee vehicle added successfully'})
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to add employee vehicle'}), 500
                    
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/employees/<plate_number>', methods=['DELETE'])
        def remove_employee(plate_number):
            """Remove an employee vehicle"""
            try:
                success = self.emp_manager.remove_employee_vehicle(plate_number)
                if success:
                    return jsonify({'status': 'success', 'message': 'Employee vehicle removed successfully'})
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to remove employee vehicle'}), 500
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/employees/check/<plate_number>', methods=['GET'])
        def check_employee(plate_number):
            """Check if a vehicle belongs to an employee"""
            try:
                is_employee = self.emp_manager.is_employee_vehicle(plate_number)
                return jsonify({'status': 'success', 'is_employee': is_employee, 'plate_number': plate_number})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _setup_socket_events(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Client connected')
            emit('status', {'message': 'Connected to vehicle monitoring system'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Client disconnected')
        
        @self.socketio.on('request_stats')
        def handle_stats_request():
            stats = self._get_combined_stats()
            emit('stats_update', stats)
    
    def _get_combined_stats(self):
        """Get combined statistics from both cameras"""
        combined_stats = {
            'total_count': 0,
            'vehicle_counts': {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0},
            'active_tracks': 0,
            'timestamp': datetime.now().isoformat(),
            'entry_exit_log': [],
            'license_plates': []
        }
        
        # Get stats from entry camera if available
        if self.vehicle_counter_entry and hasattr(self.vehicle_counter_entry, 'latest_stats'):
            entry_stats = self.vehicle_counter_entry.latest_stats
            combined_stats['total_count'] += entry_stats.get('total_count', 0)
            combined_stats['active_tracks'] += entry_stats.get('active_tracks', 0)
            
            # Combine vehicle counts
            for vehicle_type in combined_stats['vehicle_counts']:
                combined_stats['vehicle_counts'][vehicle_type] += entry_stats.get('vehicle_counts', {}).get(vehicle_type, 0)
            
            # Combine license plates
            if 'license_plates' in entry_stats:
                combined_stats['license_plates'].extend(entry_stats['license_plates'])
        
        # Get stats from exit camera if available
        if self.vehicle_counter_exit and hasattr(self.vehicle_counter_exit, 'latest_stats'):
            exit_stats = self.vehicle_counter_exit.latest_stats
            combined_stats['total_count'] += exit_stats.get('total_count', 0)
            combined_stats['active_tracks'] += exit_stats.get('active_tracks', 0)
            
            # Combine vehicle counts
            for vehicle_type in combined_stats['vehicle_counts']:
                combined_stats['vehicle_counts'][vehicle_type] += exit_stats.get('vehicle_counts', {}).get(vehicle_type, 0)
            
            # Combine license plates
            if 'license_plates' in exit_stats:
                combined_stats['license_plates'].extend(exit_stats['license_plates'])
        
        return combined_stats
    
    def _generate_frames(self, camera_id='1'):
        """Generate video frames for streaming"""
        logger.info(f"🔍 Starting video feed generation for camera {camera_id}...")
        
        # Determine which camera to use based on ID
        if camera_id == '1':
            vehicle_counter = self.vehicle_counter_entry
            camera_source = self.camera_sources.get('entry', 0)
        else:  # camera_id == '2'
            vehicle_counter = self.vehicle_counter_exit
            camera_source = self.camera_sources.get('exit', 1)
        
        cap = None
        
        while True:
            if not self.is_processing:
                # Show placeholder when not monitoring
                placeholder = self._create_placeholder_frame()
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.5)
                continue
            
            # Initialize camera when monitoring starts
            if cap is None:
                logger.info(f"🎬 Opening camera {camera_source} for monitoring...")
                cap = cv2.VideoCapture(camera_source)
                if cap.isOpened():
                    # Test if camera actually works
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"✅ Camera {camera_source} opened successfully")
                        # Set camera properties for better performance
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                    else:
                        logger.warning(f"⚠️ Camera {camera_source} opened but can't read frames")
                        cap.release()
                        cap = None
                else:
                    logger.warning(f"❌ Camera {camera_source} failed to open")
                    cap = None
                
                # If no camera found, try test videos
                if cap is None:
                    logger.info("📹 Trying vehicle test video...")
                    cap = cv2.VideoCapture("vehicle_test_video.mp4")
                    if cap.isOpened():
                        logger.info("✅ Vehicle test video opened successfully")
                    else:
                        logger.info("📹 Trying basic test video...")
                        cap = cv2.VideoCapture("test_video.mp4")
                        if cap.isOpened():
                            logger.info("✅ Basic test video opened successfully")
                
                # If still no source, create demo frames
                if cap is None or not cap.isOpened():
                    logger.warning("⚠️ No camera or video source available, using demo mode")
                    cap = None
            
            # Generate frames
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame, releasing camera")
                    cap.release()
                    cap = None
                    continue
                
                # Process frame with AI detection
                if vehicle_counter is not None:
                    try:
                        processed_frame, stats = vehicle_counter.process_frame_for_web(frame)
                        # Emit stats update
                        combined_stats = self._get_combined_stats()
                        self.socketio.emit('stats_update', combined_stats)
                    except Exception as e:
                        logger.error(f"❌ Error processing frame: {e}")
                        processed_frame = frame.copy()
                        cv2.putText(processed_frame, "Processing Error", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    processed_frame = frame.copy()
                    cv2.putText(processed_frame, f"Live Camera Feed {camera_id}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            else:
                # Demo mode - create synthetic frame
                processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                processed_frame[:] = (40, 60, 80)
                cv2.putText(processed_frame, f"DEMO MODE - No Camera {camera_id} Detected", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(processed_frame, "AI Vehicle Detection Active", 
                           (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", 
                           (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Generate demo stats
                if vehicle_counter is not None:
                    try:
                        _, stats = vehicle_counter.process_frame_for_web(processed_frame)
                        # Emit stats update
                        combined_stats = self._get_combined_stats()
                        self.socketio.emit('stats_update', combined_stats)
                    except Exception as e:
                        logger.error(f"❌ Error in demo processing: {e}")
            
            # Encode and yield frame
            try:
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"❌ Error encoding frame: {e}")
            
            time.sleep(0.033)  # ~30 FPS
            
            # Clean up when monitoring stops
            if not self.is_processing and cap is not None:
                cap.release()
                cap = None
                logger.info("📹 Camera released")
    
    def _create_placeholder_frame(self):
        """Create a placeholder frame when no monitoring is active"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        text = "Click 'Start Monitoring' to begin"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
        return frame
    
    def _create_error_frame(self, error_message):
        """Create an error frame with message"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 80)  # Dark blue background
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (100, 100, 255)
        thickness = 2
        
        # Split long messages into multiple lines
        words = error_message.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) < 35:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        # Draw each line
        y_start = frame.shape[0] // 2 - (len(lines) * 25) // 2
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            x = (frame.shape[1] - text_size[0]) // 2
            y = y_start + i * 30
            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness)
        
        return frame
    
    def run(self):
        """Run the dashboard application"""
        logger.info("Starting Vehicle Monitoring Web Dashboard...")
        logger.info(f"Access the dashboard at: http://{self.host}:{self.port}")
        
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug, allow_unsafe_werkzeug=True)