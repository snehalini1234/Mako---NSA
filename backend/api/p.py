from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import io
import logging
import schedule
import time
import threading
import requests # Added missing top-level import for batch processing

# NOTE: The 'shark_detection_ml' module (SharkTrackingPipeline class) is assumed 
# to be defined elsewhere and is required for this application to run.
from shark_detection_ml import SharkTrackingPipeline 

# Initialize Flask app
app = Flask(__name__) # Fixed: _name_ to __name__
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Fixed: _name_ to __name__

# Initialize ML pipeline
# NOTE: This requires environment variables MODEL_PATH, NASA_API_KEY, and 
# SENTINEL_HUB_CLIENT_ID to be set for the application to function correctly.
try:
    pipeline = SharkTrackingPipeline(
        model_path=os.getenv('MODEL_PATH'),
        api_keys={
            'nasa': os.getenv('NASA_API_KEY'),
            'sentinel': os.getenv('SENTINEL_HUB_CLIENT_ID')
        }
    )
    logger.info("ML Pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize SharkTrackingPipeline: {e}")
    pipeline = None # Set to None if initialization fails

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Returns the operational status of the service and model readiness."""
    return jsonify({
        'status': 'healthy',
        'service': 'Sharks from Space ML Service',
        'version': '1.0.0',
        # Safely check if the pipeline and model are loaded
        'model_loaded': pipeline is not None and getattr(pipeline, 'detection_model', None) is not None
    }), 200


# Process single image endpoint
@app.route('/api/detect', methods=['POST'])
def detect_shark():
    """
    Detect sharks in uploaded image (multipart/form-data).
    Expects: 'image' file, 'latitude', 'longitude'.
    """
    if pipeline is None:
        return jsonify({'success': False, 'error': 'ML pipeline is not initialized'}), 503

    try:
        # 1. Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']

        # 2. Parse and validate coordinates
        try:
            latitude = float(request.form.get('latitude'))
            longitude = float(request.form.get('longitude'))
        except (ValueError, TypeError):
            # Handles cases where lat/lon is missing (None) or non-numeric
            return jsonify({
                'success': False,
                'error': 'Invalid or missing latitude/longitude parameters'
            }), 400
            
        # 3. Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # 4. Run detection
        result = pipeline.process_satellite_image(
            image_array,
            latitude,
            longitude
        )
        
        if result:
            return jsonify({
                'success': True,
                'data': result
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Detection failed or no sharks found'
            }), 200 # Often 200 with an empty result is preferred for "no results"
            
    except Exception as e:
        logger.error(f"Error in detect_shark: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error during detection: {str(e)}'
        }), 500


# Batch processing endpoint
@app.route('/api/detect/batch', methods=['POST'])
def detect_batch():
    """
    Process multiple images in batch, expects JSON payload with URLs and coordinates.
    """
    if pipeline is None:
        return jsonify({'success': False, 'error': 'ML pipeline is not initialized'}), 503

    try:
        data = request.get_json()
        image_urls = data.get('image_urls', [])
        coordinates = data.get('coordinates', [])
        
        if len(image_urls) != len(coordinates):
            return jsonify({
                'success': False,
                'error': 'Number of images and coordinates must match'
            }), 400
            
        results = []
        for i, (url, coords) in enumerate(zip(image_urls, coordinates)):
            # Ensure coordinates are in the expected (lat, lon) format and convertible
            try:
                lat, lon = float(coords[0]), float(coords[1])
            except (ValueError, TypeError, IndexError):
                logger.warning(f"Skipping batch item {i} due to invalid coordinates: {coords}")
                continue

            try:
                # Download and process image
                response = requests.get(url, timeout=10)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                image = Image.open(io.BytesIO(response.content))
                image_array = np.array(image)
                
                result = pipeline.process_satellite_image(
                    image_array,
                    lat,
                    lon
                )
                
                if result:
                    results.append(result)
            except requests.exceptions.RequestException as req_e:
                logger.error(f"Error downloading or connecting to {url}: {str(req_e)}")
            except Exception as e:
                logger.error(f"Error processing image from {url}: {str(e)}")
                continue
            
        return jsonify({
            'success': True,
            'count': len(results),
            'data': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in detect_batch: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Get environmental data endpoint
@app.route('/api/environmental-data', methods=['GET'])
def get_environmental_data():
    """
    Get environmental data for a location.
    Query params: latitude, longitude.
    """
    if pipeline is None:
        return jsonify({'success': False, 'error': 'ML pipeline is not initialized'}), 503

    try:
        # Validate coordinates
        try:
            latitude = float(request.args.get('latitude'))
            longitude = float(request.args.get('longitude'))
        except (ValueError, TypeError):
            # Handles cases where lat/lon is missing (None) or non-numeric
            return jsonify({
                'success': False,
                'error': 'Missing or invalid latitude/longitude query parameters'
            }), 400
            
        env_data = pipeline.data_processor.get_environmental_data(
            latitude,
            longitude
        )
        
        return jsonify({
            'success': True,
            'data': env_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting environmental data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Model information endpoint
@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the ML model"""
    try:
        # Assuming the pipeline and model structure is stable for this endpoint
        model_size = getattr(getattr(pipeline, 'detection_model', None), 'img_size', 'N/A')
        confidence = getattr(getattr(pipeline, 'detection_model', None), 'confidence_threshold', 'N/A')

        return jsonify({
            'success': True,
            'data': {
                'model_type': 'CNN-ResNet50',
                'version': '1.0',
                'input_size': model_size,
                'confidence_threshold': confidence,
                'species_detected': [
                    'Great White', 'Tiger Shark', 'Bull Shark', 'Hammerhead',
                    'Blue Shark', 'Mako Shark', 'Whale Shark', 'Basking Shark',
                    'Thresher Shark', 'Lemon Shark', 'Nurse Shark', 'Reef Shark',
                    'Goblin Shark', 'Sand Tiger', 'Unknown'
                ]
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Scheduled task to process satellite imagery
def scheduled_satellite_processing():
    """
    Background task to automatically process new satellite imagery.
    Runs every hour.
    """
    if pipeline is None:
        logger.warning("Scheduled processing skipped: ML pipeline is not initialized.")
        return
        
    logger.info("Starting scheduled satellite processing...")
    
    try:
        # In a real-world scenario, this function would handle:
        # 1. Querying for new imagery using the Sentinel/NASA API wrappers in the pipeline.
        # 2. Downloading images in predefined monitoring areas.
        # 3. Running pipeline.process_satellite_image(...) for each one.
        # 4. Storing the detection results in a persistent database (e.g., Firestore/PostgreSQL).
        
        logger.info("Scheduled processing completed - Placeholder functionality executed.")
        
    except Exception as e:
        logger.error(f"Error in scheduled processing: {str(e)}")


# Schedule the task
schedule.every(1).hour.do(scheduled_satellite_processing)

def run_scheduler():
    """Run scheduled tasks in background thread."""
    logger.info("Scheduler thread started.")
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Error running scheduled task: {e}")
        time.sleep(60)


# Start scheduler in background thread
# Daemon=True ensures the thread exits when the main program exits
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    # Log the full exception for better debugging
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred. Check server logs for details.'
    }), 500


if __name__ == '__main__': # Fixed: _name_ to __name__
    # Get port from environment or use default
    port = int(os.getenv('PORT', 8000))
    
    # Run Flask app
    logger.info(f"Starting ML Service on port {port}")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('DEBUG', 'False').lower() == 'true',
        # Do not use threading/reloader with the custom scheduler thread
        threaded=True, 
        use_reloader=os.getenv('DEBUG', 'False').lower() == 'true'
    )