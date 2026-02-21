"""
FoodSight AI - Flask Backend Server
====================================

This is the main backend server for the FoodSight AI web application.
It serves the trained MobileNetV2 model and provides REST API endpoints
for food classification.

Author: FoodSight AI Team
Date: February 2026
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import logging
from datetime import datetime
import shutil
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import uuid
import magic

# Load environment variables
load_dotenv()

from backend.nutrition_data import NUTRITION_DATABASE, get_nutrition_info, get_all_dishes
from backend.dataset_config import EXPANDED_CLASS_NAMES, DATASET_STATS, CATEGORY_MAPPING, REGION_MAPPING

# New imports for Auth and DB
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='frontend/static', template_folder='frontend/templates')
CORS(app)  # Enable CORS for frontend communication

# Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 5 * 1024 * 1024))
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
# Use absolute path for SQLite
default_db = 'sqlite:///' + os.path.join(basedir, 'instance', 'users.db')
db_url = os.getenv('DATABASE_URL', default_db)

# Ensure the database URL is absolute for SQLite
if db_url.startswith('sqlite:///'):
    db_path = db_url.replace('sqlite:///', '')
    if not os.path.isabs(db_path):
        db_url = 'sqlite:///' + os.path.abspath(os.path.join(basedir, db_path))

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize Extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Rate Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    # Relationship to history
    history = db.relationship('ScanHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

# Scan History Model
class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    dish_name = db.Column(db.String(100), nullable=False)
    calories = db.Column(db.Integer)
    image_url = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'dish_name': self.dish_name,
            'calories': self.calories,
            'image_url': self.image_url,
            'timestamp': self.timestamp.isoformat()
        }

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Food class names
# Try to load dynamically from training results first
CLASS_NAMES = []

def load_class_names():
    """Load class names from the trained model's artifact file."""
    global CLASS_NAMES
    
    # Paths to check
    paths_to_check = [
        os.path.join('..', '02_Results_And_Evaluation', 'class_names.txt'),
        os.path.join('02_Results_And_Evaluation', 'class_names.txt'),
        'class_names.txt',
        os.path.join('scripts', 'class_names.txt'),
        os.path.join('backend', 'class_names.txt')
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"[OK] Loaded {len(CLASS_NAMES)} classes from {path}")
                return
            except Exception as e:
                logger.error(f"Failed to load class names from {path}: {e}")

    # Fallback to dataset directory scanning
    dataset_path = os.path.join('dataset', 'train')
    if not CLASS_NAMES and os.path.exists(dataset_path):
        try:
            CLASS_NAMES = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
            logger.info(f"[OK] Loaded {len(CLASS_NAMES)} classes from dataset directory")
            return
        except Exception as e:
            logger.error(f"Failed to scan dataset directory: {e}")

    # Final fallback to config file
    if not CLASS_NAMES:
        logger.warning("[WARNING] Could not load dynamic classes. Falling back to dataset_config.py")
        CLASS_NAMES = EXPANDED_CLASS_NAMES

# Load classes on module import
load_class_names()

# Flag for compatibility
USE_EXPANDED_DATASET = True

# Global model variable
model = None

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)
    os.makedirs(os.path.join('frontend', 'static', 'samples'), exist_ok=True)
    logger.info("[OK] Directories created/verified")

def load_trained_model():
    """
    Load the trained MobileNetV3Large model (TensorFlow/Keras).
    Searches for the fine-tuned model first, then falls back to the base model.
    """
    global model
    
    # Search for model files in order of preference
    model_search_paths = [
        os.path.join('model', 'best_model_v6_final (fine-tuned).keras'),  # Fine-tuned (best)
        os.path.join('model', 'best_model_v6_final.keras'),                # Base trained model
    ]
    
    for model_path in model_search_paths:
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading model from {model_path}...")
                model = tf.keras.models.load_model(model_path)
                logger.info(f"[OK] Model loaded successfully from {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                continue
    
    logger.error("[ERROR] No model file found! Expected one of: " + ", ".join(model_search_paths))
    return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path):
    """
    Preprocess image for TensorFlow model prediction.
    Optimized for single-inference speed.
    """
    try:
        # Load and resize using Keras utility
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create batch axis
        
        # Normalize/Scale if necessary (MobileNetV3 handles this internally 
        # but we follow standard practice)
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Too Many Requests',
        'message': 'Slow down! You are making too many requests. Please try again later.'
    }), 429

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({
        'error': 'File Too Large',
        'message': f'The file exceeds the maximum allowed size of {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB.'
    }), 413

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify server and model status.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0 (TensorFlow)'
    })

@app.route('/api/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    """
    Main prediction endpoint for food classification.
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The AI model is not available. Please contact support.'
            }), 503
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please upload an image file.'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'message': 'Please select an image file to upload.'}), 400
        
        # Secure File Validation
        # 1. MIME check via Magic Bytes
        # Note: 'magic' library needs to be imported (e.g., `import magic`)
        # For this example, assuming `magic` is available or this is a placeholder.
        # If `magic` is not available, this part will cause an error.
        try:
            import magic
            header = file.read(1024)
            file.seek(0)
            mime = magic.from_buffer(header, mime=True)
            if mime not in ['image/jpeg', 'image/png', 'image/webp']:
                return jsonify({'error': f'Invalid file type: {mime}. Only JPG, PNG, WEBP allowed.'}), 400
        except ImportError:
            logger.warning("Python 'magic' library not found. Skipping MIME type validation.")
        except Exception as e:
            logger.error(f"Error during MIME type validation: {e}")
            return jsonify({'error': 'File validation failed', 'message': 'Could not validate file type.'}), 500

        if file and allowed_file(file.filename):
            # 2. Filename Sanitization with UUID
            # Note: 'uuid' library needs to be imported (e.g., `import uuid`)
            # For this example, assuming `uuid` is available.
            import uuid
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            
            logger.info(f"Processing image: {filename}")
            
            # Preprocess and Predict
            input_batch = preprocess_image(filepath)
            
            # --- PERFORMANCE OPTIMIZATION ---
            # Using model(input, training=False) is faster for single-item inference
            # than model.predict() which has overhead for batch processing and logging.
            predictions = model(input_batch, training=False)
            probabilities = predictions.numpy()[0]

            predicted_idx = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(probabilities[predicted_idx] * 100)

                
            # Get top 3 predictions
            # TensorFlow/Numpy logic
            top3_idx = probabilities.argsort()[-3:][::-1]
            top3_prob = probabilities[top3_idx]
            
            top_3 = []
            for i in range(len(top3_idx)):
                idx = top3_idx[i]
                top_3.append({
                    'class': CLASS_NAMES[idx],
                    'confidence': float(top3_prob[i] * 100)
                })
            
            # Create all predictions dictionary
            all_predictions = {
                CLASS_NAMES[i]: float(probabilities[i] * 100)
                for i in range(len(CLASS_NAMES))
            }
            
            logger.info(f"Prediction: {predicted_class} ({confidence:.2f}%)")
            
            # Clean up uploaded file (optional - comment out to keep files)
            try:
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Error removing file {filepath}: {e}")
            
            # Get portion size (default to medium)
            portion = request.form.get('portion', 'medium')
            
            # Get nutrition information for predicted dish with portion
            nutrition_info = get_nutrition_info(predicted_class, portion)
            
            # Return results with nutrition data
            response_data = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'top_3': top_3,
                'all_predictions': all_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add nutrition info if available
            if nutrition_info:
                from backend.nutrition_data import get_health_indicators, calculate_health_score, get_dietary_suitability
                response_data['nutrition'] = nutrition_info
                # Add health indicators, score and suitability
                response_data['health_indicators'] = get_health_indicators(nutrition_info)
                response_data['health_score'] = calculate_health_score(nutrition_info)
                response_data['suitability'] = get_dietary_suitability(nutrition_info)
            
            return jsonify(response_data)
        else:
            return jsonify({
                'error': 'Invalid file type',
                'message': f'Allowed file types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        probabilities = predictions[0]

        predicted_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx] * 100)

            
        # Get top 3 predictions
        # TensorFlow/Numpy logic
        top3_idx = probabilities.argsort()[-3:][::-1]
        top3_prob = probabilities[top3_idx]
        
        top_3 = []
        for i in range(len(top3_idx)):
            idx = top3_idx[i]
            top_3.append({
                'class': CLASS_NAMES[idx],
                'confidence': float(top3_prob[i] * 100)
            })
        
        # Create all predictions dictionary
        all_predictions = {
            CLASS_NAMES[i]: float(probabilities[i] * 100)
            for i in range(len(CLASS_NAMES))
        }
        
        logger.info(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        
        # Clean up uploaded file (optional - comment out to keep files)
        try:
            os.remove(filepath)
        except:
            pass
        
        # Get portion size (default to medium)
        portion = request.form.get('portion', 'medium')
        
        # Get nutrition information for predicted dish with portion
        nutrition_info = get_nutrition_info(predicted_class, portion)
        
        # Return results with nutrition data
        response_data = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'top_3': top_3,
            'all_predictions': all_predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add nutrition info if available
        if nutrition_info:
            response_data['nutrition'] = nutrition_info
            # Add health indicators, score and suitability
            from backend.nutrition_data import get_health_indicators, calculate_health_score, get_dietary_suitability
            response_data['health_indicators'] = get_health_indicators(nutrition_info)
            response_data['health_score'] = calculate_health_score(nutrition_info)
            response_data['suitability'] = get_dietary_suitability(nutrition_info)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Prediction failed',
            'message': 'An error occurred while processing your image. Please try again.'
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """
    Get list of all food classes the model can recognize.
    """
    return jsonify({
        'classes': CLASS_NAMES,
        'count': len(CLASS_NAMES)
    })

@app.route('/static/samples/<path:filename>')
def serve_sample(filename):
    """Serve sample images"""
    return send_from_directory(os.path.join('frontend', 'static', 'samples'), filename)

@app.route('/api/nutrition/<dish_name>', methods=['GET'])
def get_nutrition(dish_name):
    """
    Get nutrition information for a specific dish.
    """
    # Decode and normalize dish name
    from urllib.parse import unquote
    dish_name = unquote(dish_name)
    
    # Get portion size from query params
    portion = request.args.get('portion', 'medium')
    quantity = request.args.get('q')
    unit = request.args.get('u')
    
    nutrition_info = get_nutrition_info(dish_name, quantity=quantity, unit=unit, portion=portion)
    
    if nutrition_info:
        from backend.nutrition_data import get_health_indicators, calculate_health_score, get_dietary_suitability
        return jsonify({
            'success': True,
            'dish_name': dish_name,
            'nutrition': nutrition_info,
            'health_indicators': get_health_indicators(nutrition_info),
            'health_score': calculate_health_score(nutrition_info),
            'suitability': get_dietary_suitability(nutrition_info)
        })
    else:
        return jsonify({
            'error': 'Dish not found',
            'message': f'No nutrition information available for "{dish_name}"'
        }), 404

@app.route('/api/nutrition/all', methods=['GET'])
def get_all_nutrition():
    """
    Get nutrition information for all dishes.
    """
    return jsonify({
        'success': True,
        'total_dishes': len(NUTRITION_DATABASE),
        'dishes': NUTRITION_DATABASE
    })

@app.route('/api/dataset/info', methods=['GET'])
def get_dataset_info():
    """
    Get information about the expanded dataset.
    """
    return jsonify({
        'success': True,
        'current_model': {
            'classes': CLASS_NAMES,
            'total_classes': len(CLASS_NAMES),
            'status': 'active'
        },
        'expanded_dataset': {
            'classes': EXPANDED_CLASS_NAMES,
            'total_classes': len(EXPANDED_CLASS_NAMES),
            'status': 'ready_for_training',
            'statistics': DATASET_STATS
        },
        'categories': list(CATEGORY_MAPPING.keys()),
        'regions': list(REGION_MAPPING.keys())
    })

@app.route('/api/dishes/category/<category>', methods=['GET'])
def get_dishes_by_category(category):
    """
    Get all dishes in a specific category.
    """
    from urllib.parse import unquote
    category = unquote(category)
    
    dishes = CATEGORY_MAPPING.get(category, [])
    
    if dishes:
        return jsonify({
            'success': True,
            'category': category,
            'dishes': dishes,
            'count': len(dishes)
        })
    else:
        return jsonify({
            'error': 'Category not found',
            'message': f'No dishes found for category "{category}"',
            'available_categories': list(CATEGORY_MAPPING.keys())
        }), 404

@app.route('/api/dishes/region/<region>', methods=['GET'])
def get_dishes_by_region(region):
    """
    Get all dishes from a specific region.
    """
    from urllib.parse import unquote
    region = unquote(region)
    
    dishes = REGION_MAPPING.get(region, [])
    
    if dishes:
        return jsonify({
            'success': True,
            'region': region,
            'dishes': dishes,
            'count': len(dishes)
        })
    else:
        return jsonify({
            'error': 'Region not found',
            'message': f'No dishes found for region "{region}"',
            'available_regions': list(REGION_MAPPING.keys())
        }), 404

# ============================================================================
# SEARCH ENDPOINT
# ============================================================================

@app.route('/api/search', methods=['GET'])
@limiter.limit("10 per minute")
def search_food():
    """
    Search for a food item in the nutrition database.
    """
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Search in NUTRITION_DATABASE
    results = []
    query_lower = query.lower()
    
    # Exact match first
    if query in NUTRITION_DATABASE:
        results.append(query)
    else:
        # Partial match
        for dish in NUTRITION_DATABASE.keys():
            if query_lower in dish.lower():
                results.append(dish)

    if not results:
        return jsonify({
            'success': False,
            'message': 'This food item is not in our database yet. Please wait for future updates!',
            'query': query
        })

    # For now, return the best match if multiple found, or just the first one
    best_match = results[0]
    portion = request.args.get('portion', 'medium')
    nutrition_info = get_nutrition_info(best_match, portion)

    if nutrition_info:
        from backend.nutrition_data import get_health_indicators, calculate_health_score, get_dietary_suitability
        return jsonify({
            'success': True,
            'predicted_class': best_match,
            'confidence': 100.0, # Search match is 100% "confident" that it's that dish
            'nutrition': nutrition_info,
            'health_indicators': get_health_indicators(nutrition_info),
            'health_score': calculate_health_score(nutrition_info),
            'suitability': get_dietary_suitability(nutrition_info),
            'top_3': [{'class': best_match, 'confidence': 100.0}],
            'is_search': True
        })
    
    return jsonify({
        'success': False,
        'message': 'Failed to retrieve nutrition info for match.',
        'query': query
    })

# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'username': current_user.username
        })
    return jsonify({'authenticated': False})

@app.route('/api/auth/signup', methods=['POST'])
@limiter.limit("5 per minute")
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Missing fields'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'User already exists'}), 400

    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'success': True, 'message': 'User created successfully'})

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify({'success': True, 'username': user.username})
    
    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/api/auth/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

# History Endpoints
@app.route('/api/history/save', methods=['POST'])
@login_required
def save_history():
    data = request.json
    dish_name = data.get('dish_name')
    calories = data.get('calories')
    # image_url removed for privacy as requested

    if not dish_name:
        return jsonify({'error': 'Missing dish name'}), 400

    new_entry = ScanHistory(
        user_id=current_user.id,
        dish_name=dish_name,
        calories=calories
        # image_url=None
    )
    db.session.add(new_entry)
    db.session.commit()

    return jsonify({'success': True, 'message': 'History saved'})

@app.route('/api/history', methods=['DELETE'])
@login_required
def clear_history():
    """Clear all scan history for the current user"""
    ScanHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'success': True, 'message': 'History cleared successfully'})

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    history = ScanHistory.query.filter_by(user_id=current_user.id).order_by(ScanHistory.timestamp.desc()).all()
    return jsonify({
        'success': True,
        'history': [entry.to_dict() for entry in history]
    })

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

def initialize_app():
    """Initialize the application on startup"""
    logger.info("=" * 60)
    logger.info("FoodSight AI - Starting Application")
    logger.info("=" * 60)
    
    # Create necessary directories
    create_directories()

    # Initialize Database
    with app.app_context():
        db.create_all()
        logger.info("[OK] Database initialized")
    
    # Load the trained model
    if not load_trained_model():
        logger.warning("⚠ Application started but model is not loaded!")
        logger.warning("⚠ Predictions will not work until model is available.")
    
    logger.info("=" * 60)
    logger.info("✓ Application ready!")
    logger.info("=" * 60)

# Initialize on import
initialize_app()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("FoodSight AI - Web Application")
    print("=" * 60)
    print("\nServer starting at: http://localhost:5001")
    print("API Health Check: http://localhost:5001/api/health")
    print("\nPress Ctrl+C to stop the server\n")
    print("=" * 60 + "\n")
    
    # Run Flask development server
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        use_reloader=True  # Enable reloading for development
    )
