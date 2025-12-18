"""
Cattle Breed Classification Web Application
Backend Server using Flask and TensorFlow/Keras
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = '/Users/krutikakatke/Documents/classification model/website/efficientnetv2-b0_phase3_final.keras'

# Cattle breed class names (REPLACE THIS LIST with your actual breed names)
CLASS_NAMES = [
     "Amritmahal",
    "Ayrshire",
    "bachaur",
    "badri",
    "Bargur",
    "bhelai",
    "dagri",
    "Dangi",
    "Deoni",
    "gangatari",
    "gaolao",
    "ghumsari",
    "Gir",
    "Hallikar",
    "Hariana",
    "Himachali Pahari",
    "Kangayam",
    "Kankrej",
    "Kenkatha",
    "Khariar",
    "kherigarh",
    "Khillari",
    "Konkan Kapila",
    "Kosali",
    "Krishna_Valley",
    "Ladakhi",
    "Lakhimi",
    "Malnad_gidda",
    "malvi",
    "Mewati",
    "motu",
    "nagori",
    "Nari",
    "Nimari",
    "Ongole",
    "Poda Thirupu",
    "ponwar",
    "Pulikulam",
    "Punganur",
    "Purnea",
    "Rathi",
    "Red kandhari",
    "Red_Sindhi",
    "Sahiwal",
    "Shweta Kapila",
    "siri",
    "Tharparkar",
    "thutho",
    "Umblachery",
    "Vechur"
]

# Global variable to store loaded model
model = None


def load_model():
    """Load the trained Keras model"""
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img):
    """
    Preprocess image for EfficientNetV2-B0 inference
    
    Args:
        img: PIL Image object
        
    Returns:
        preprocessed_img: numpy array ready for model prediction
    """
    # Resize to 224x224 (EfficientNetV2-B0 input size)
    img = img.resize((224, 224))
    
    # Convert to RGB if image has different mode (e.g., RGBA, grayscale)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert PIL image to numpy array
    img_array = image.img_to_array(img)
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply EfficientNetV2 preprocessing
    # This normalizes pixel values to [-1, 1] range
    preprocessed_img = preprocess_input(img_array)
    
    return preprocessed_img


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Read and open image
        img = Image.open(file.stream)
        
        # Preprocess image
        preprocessed_img = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(preprocessed_img, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        predicted_breed = CLASS_NAMES[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'breed': CLASS_NAMES[idx],
                'confidence': float(predictions[0][idx]) * 100
            }
            for idx in top_3_indices
        ]
        
        # Convert image to base64 for display
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'predicted_breed': predicted_breed,
            'confidence': round(confidence, 2),
            'top_predictions': top_3_predictions,
            'image': img_str
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy', 'model_loaded': True})


if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        print("ERROR: Failed to load model. Please ensure 'final_locked_model.keras' exists.")
        print("Exiting...")
        exit(1)
    
    print("\n" + "="*60)
    print("🐄 CATTLE BREED CLASSIFIER - SERVER STARTING")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Number of breeds: {len(CLASS_NAMES)}")
    print(f"Server: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)