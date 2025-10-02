# shark_detection_ml.py - Machine Learning module for shark detection from satellite imagery

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import requests
from datetime import datetime
import json
import os
from PIL import Image
# rasterio and rasterio.windows imported but not used in the current version.
# They are placeholders for more advanced geospatial data handling (e.g., reading TIFF files).
# import rasterio  # Removed because it is not used and causes import error

class SharkDetectionModel:
    """
    AI model for detecting sharks from satellite imagery
    Uses CNN architecture optimized for marine life detection
    """
    
    # FIX: Corrected constructor name from _init_ to __init__
    def __init__(self, model_path=None):
        self.model = None
        self.img_size = (224, 224)
        self.confidence_threshold = 0.75
        
        # Check if TensorFlow is available before proceeding (good practice)
        if not tf.test.is_built_with_cuda() and not tf.test.is_gpu_available():
             print("Warning: TensorFlow is running without GPU support. Performance may be slow.")

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build CNN model for shark detection using ResNet50 base"""
        
        # Ensure ResNet50 input is normalized correctly (will be handled in preprocessing)
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification layers
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layers for multi-task learning (species classification + confidence score)
        species_output = layers.Dense(15, activation='softmax', name='species')(x)
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        self.model = keras.Model(
            inputs=inputs,
            outputs=[species_output, confidence_output]
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'species': 'categorical_crossentropy',
                'confidence': 'binary_crossentropy'
            },
            metrics={
                'species': ['accuracy'],
                'confidence': ['accuracy']
            }
        )
        
        print("Model built successfully")
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path):
        """Save trained model"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def preprocess_image(self, image):
        """
        Preprocess satellite image for model input.
        Includes resizing, contrast enhancement (CLAHE), and normalization.
        """
        
        # 1. Resize image
        image = cv2.resize(image, self.img_size)
        
        # 2. FIX: Apply CLAHE (Contrast Enhancement) on 8-bit image data
        # CLAHE expects 8-bit single-channel input. Convert to LAB and apply on L channel.
        if image.dtype != np.uint8:
             # Convert to 8-bit if the image came from a float source (e.g., NumPy random array)
             image = (image * 255).astype(np.uint8) 

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # 3. Normalize pixel values for model input (0.0 to 1.0)
        image = image.astype('float32') / 255.0
        
        return image
    
    def detect_sharks(self, image):
        """Detect sharks in satellite image"""
        
        # Ensure model is loaded/built
        if self.model is None:
             raise RuntimeError("Model is not loaded or built.")

        # Preprocess
        processed_img = self.preprocess_image(image)
        # Add batch dimension
        input_batch = np.expand_dims(processed_img, axis=0)
        
        # Predict
        species_pred, confidence_pred = self.model.predict(input_batch)
        
        species_classes = [
            'Great White', 'Tiger Shark', 'Bull Shark', 'Hammerhead',
            'Blue Shark', 'Mako Shark', 'Whale Shark', 'Basking Shark',
            'Thresher Shark', 'Lemon Shark', 'Nurse Shark', 'Reef Shark',
            'Goblin Shark', 'Sand Tiger', 'Unknown'
        ]
        
        species_idx = np.argmax(species_pred[0])
        species = species_classes[species_idx]
        species_confidence = float(species_pred[0][species_idx])
        detection_confidence = float(confidence_pred[0][0])
        
        return {
            'species': species,
            'species_confidence': species_confidence,
            'detection_confidence': detection_confidence,
            'overall_confidence': (species_confidence + detection_confidence) / 2
        }


class SatelliteDataProcessor:
    """
    Process satellite data from various sources (NASA, ESA, etc.)
    Extract environmental parameters relevant to shark detection
    """
    
    # FIX: Corrected constructor name from _init_ to __init__
    def __init__(self, api_keys=None):
        self.api_keys = api_keys or {}
        self.nasa_api_key = self.api_keys.get('nasa', 'DEMO_KEY')
    
    def fetch_nasa_data(self, latitude, longitude, date=None):
        """Fetch environmental data from NASA APIs"""
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # NASA POWER API for environmental data
        # Note: This API primarily provides atmospheric/meteorological data,
        # not typically marine parameters like SST or Chlorophyll.
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        params = {
            'parameters': 'T2M,RH2M,PRECTOTCORR',  # Temperature (at 2m), Humidity, Precipitation
            'community': 'AG',
            'longitude': longitude,
            'latitude': latitude,
            'start': date,
            'end': date,
            'format': 'JSON'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10) # Added timeout
            response.raise_for_status()
            # In a real scenario, you would parse the T2M (air temp) and potentially use it.
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching NASA data: {e}")
            return None
    
    def calculate_sea_surface_temp(self, latitude, longitude):
        """Calculate sea surface temperature from satellite data (SIMULATED)"""
        
        # Simulated SST calculation (in production, use actual satellite data)
        base_temp = 20.0  # Base temperature in Celsius
        lat_factor = abs(latitude) / 90  # Latitude effect
        # Simple seasonal factor
        seasonal_factor = np.sin((datetime.now().month / 12) * 2 * np.pi)
        
        sst = base_temp - (lat_factor * 15) + (seasonal_factor * 5)
        
        return round(sst, 2)
    
    def calculate_chlorophyll_concentration(self, latitude, longitude):
        """Estimate chlorophyll-a concentration (SIMULATED)"""
        
        # Simulated chlorophyll calculation
        coastal_distance = self.estimate_coastal_distance(latitude, longitude)
        
        if coastal_distance < 50:  # Near coast (higher likely)
            chlorophyll = np.random.uniform(0.5, 2.0)
        else:  # Open ocean (lower likely)
            chlorophyll = np.random.uniform(0.05, 0.3)
        
        return round(chlorophyll, 3)
    
    def estimate_coastal_distance(self, latitude, longitude):
        """Estimate distance to nearest coast (simplified)"""
        # Simplified calculation - in production use actual coastal database
        # This is a very rough simulation; better to use a distance-to-coast library in production
        return abs(longitude) * 10
    
    def calculate_water_clarity(self, latitude, longitude):
        """Calculate water clarity index (based on simulated chlorophyll)"""
        
        chlorophyll = self.calculate_chlorophyll_concentration(latitude, longitude)
        
        # Lower chlorophyll generally means clearer water
        if chlorophyll < 0.1:
            clarity = 'Very High'
            clarity_value = 0.9
        elif chlorophyll < 0.3:
            clarity = 'High'
            clarity_value = 0.75
        elif chlorophyll < 0.8:
            clarity = 'Medium'
            clarity_value = 0.5
        else:
            clarity = 'Low'
            clarity_value = 0.25
        
        return {
            'clarity': clarity,
            'value': clarity_value,
            'chlorophyll': chlorophyll
        }
    
    def get_environmental_data(self, latitude, longitude):
        """Get comprehensive environmental data for a location"""
        
        sst = self.calculate_sea_surface_temp(latitude, longitude)
        water_clarity = self.calculate_water_clarity(latitude, longitude)
        
        return {
            'sea_surface_temp': sst,
            'water_clarity': water_clarity['clarity'],
            'water_clarity_value': water_clarity['value'],
            'chlorophyll_level': water_clarity['chlorophyll'],
            'salinity': round(np.random.uniform(32, 37), 1),  # Typical ocean salinity
            'location': {
                'latitude': latitude,
                'longitude': longitude
            }
        }


class SharkTrackingPipeline:
    """
    Complete pipeline for processing satellite imagery and detecting sharks
    """
    
    # FIX: Corrected constructor name from _init_ to __init__
    def __init__(self, model_path=None, api_keys=None):
        self.detection_model = SharkDetectionModel(model_path)
        self.data_processor = SatelliteDataProcessor(api_keys)
        # Using 80% confidence threshold for reporting to API
        self.report_confidence_threshold = 0.80 
        self.api_endpoint = os.getenv('API_ENDPOINT', 'http://localhost:5000/api')
    
    def process_satellite_image(self, image_path, latitude, longitude):
        """Process a single satellite image and extract shark detections"""
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                # Check for successful read
                if image is None:
                     raise FileNotFoundError(f"Could not read image from path: {image_path}")
                # OpenCV loads in BGR, convert to RGB for ML model
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            else:
                # Assume image_path is already a NumPy array (like the dummy data)
                image = image_path
                # Ensure the image array is a valid color image
                if image.ndim != 3 or image.shape[2] != 3:
                     raise ValueError("Image array must be a 3-channel color image.")
            
            # Detect sharks
            detection_result = self.detection_model.detect_sharks(image)
            
            # Get environmental data
            env_data = self.data_processor.get_environmental_data(latitude, longitude)
            
            # Combine results
            # The confidence is multiplied by 100 to convert from 0-1 float to 0-100 percentage
            result = {
                'species': detection_result['species'],
                'latitude': latitude,
                'longitude': longitude,
                'confidence': detection_result['overall_confidence'] * 100,
                'temperature': env_data['sea_surface_temp'], # Simplified: using SST as 'temperature'
                'detectedAt': datetime.now().isoformat(),
                'satelliteSource': 'Landsat-8',  # Or actual satellite source
                'environmentalData': {
                    'seaSurfaceTemp': env_data['sea_surface_temp'],
                    'chlorophyllLevel': env_data['chlorophyll_level'],
                    'waterClarity': env_data['water_clarity_value'],
                    'salinity': env_data['salinity']
                },
                'metadata': {
                    'detectionMethod': 'CNN-ResNet50',
                    'modelVersion': '1.0',
                    'processedBy': 'SharkTrackingPipeline'
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing image at ({latitude}, {longitude}): {e}")
            return None
    
    def process_batch(self, image_paths, coordinates):
        """Process multiple satellite images"""
        
        results = []
        
        # Check if coordinates and image_paths have the same length
        if len(image_paths) != len(coordinates):
             print("Warning: Image list and coordinate list lengths do not match. Skipping batch.")
             return []
             
        for img_path, (lat, lon) in zip(image_paths, coordinates):
            result = self.process_satellite_image(img_path, lat, lon)
            
            # Check confidence before adding to results
            if result and (result['confidence'] / 100) >= self.report_confidence_threshold:
                results.append(result)
            elif result:
                 print(f"Detection ignored: {result['species']} at ({lat}, {lon}) due to low confidence ({result['confidence']:.2f}%)")
        
        return results
    
    def send_to_api(self, detection_data):
        """Send detection data to backend API"""
        
        try:
            print(f"Attempting to send detection data to API: {self.api_endpoint}/detections")
            response = requests.post(
                f"{self.api_endpoint}/detections",
                json=detection_data,
                headers={'Content-Type': 'application/json'},
                timeout=5 # Added timeout
            )
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error sending to API. Is the server running at {self.api_endpoint}? Error: {e}")
            return None
    
    def run_detection_pipeline(self, image_paths, coordinates):
        """Run complete detection pipeline and send results to API"""
        
        print(f"Processing {len(image_paths)} satellite images...")
        
        results = self.process_batch(image_paths, coordinates)
        
        print(f"Found {len(results)} high-confidence shark detections (>{self.report_confidence_threshold*100}%)")
        
        # Send results to API
        for result in results:
            api_response = self.send_to_api(result)
            if api_response:
                print(f"Detection saved: {result['species']} at ({result['latitude']}, {result['longitude']}) - API Status: Success")
            else:
                 print(f"Detection FAILED to save: {result['species']} at ({result['latitude']}, {result['longitude']})")
        
        return results


# Example usage
# FIX: Corrected main block name from _name_ to __name__
if __name__ == "__main__":
    # Initialize pipeline
    # Note: This will build a dummy model as no model_path is provided.
    pipeline = SharkTrackingPipeline()
    
    # Example: Process sample images
    # In production, these would be actual satellite images retrieved from a data store
    sample_coordinates = [
        (-34.0, 18.5),  # Cape Town
        (-33.8, 18.9),  # False Bay
        (-33.5, 18.3),  # Table Bay
    ]
    
    # Create dummy images for demonstration
    sample_images = []
    for i in range(3):
        # Dummy image (512x512 RGB 8-bit image)
        dummy_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) 
        sample_images.append(dummy_img)
    
    # Run detection
    results = pipeline.run_detection_pipeline(sample_images, sample_coordinates)
    
    print("\nDetection Summary:")
    print(json.dumps(results, indent=2))