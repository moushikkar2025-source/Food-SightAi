"""
FoodSight AI - Flask Backend Server
====================================

This is the main backend server for the FoodSight AI web application.
It serves the trained MobileNetV3 model and provides REST API endpoints
for food classification.

Author: FoodSight AI Team
Date: February 2026
"""

import os
import sys
import uuid
import logging
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tensorflow as tf
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='frontend/static', template_folder='frontend/templates')
CORS(app)

# Import nutrition and config
try:
    from backend.nutrition_data import NUTRITION_DATABASE, get_nutrition_info, get_all_dishes, get_health_indicators, calculate_health_score, get_dietary_suitability
    from backend.dataset_config import EXPANDED_CLASS_NAMES, DATASET_STATS, CATEGORY_MAPPING, REGION_MAPPING
except ImportError as e:
    logger.error(f"Critical Import Error: {e}")
    raise

# Database Configuration (for Auth)
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

# Flask Config
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'}
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-key-123')

# Robust Database URI handling (SQLite vs PostgreSQL for Render)
db_url = os.getenv('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    # SQLAlchemy 1.4+ requires 'postgresql://' instead of 'postgres://'
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url or ('sqlite:///' + os.path.join(basedir, 'instance', 'users.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Rate Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1000 per day", "200 per hour"],
    storage_uri="memory://",
)

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    history = db.relationship('ScanHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    dish_name = db.Column(db.String(100), nullable=False)
    calories = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'dish_name': self.dish_name,
            'calories': self.calories,
            'timestamp': self.timestamp.isoformat()
        }

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- MODEL LOADING ---
CLASS_NAMES = []
model = None
_model_load_lock = threading.Lock()

def load_class_names():
    global CLASS_NAMES
    paths = [
        os.path.join(basedir, 'class_names.txt'),
        os.path.join(basedir, 'backend', 'class_names.txt'),
        os.path.join(basedir, 'model', 'class_names.txt')
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(CLASS_NAMES)} classes from {p}")
            return
    CLASS_NAMES = EXPANDED_CLASS_NAMES  # Fallback
    logger.warning("Using fallback class names from config")

def _load_model_sync():
    """Load model in current thread (safe for inference in same process)."""
    global model
    model_paths = [
        os.path.join(basedir, 'model', 'best_model_v6_final (fine-tuned).keras'),
        os.path.join(basedir, 'model', 'best_model_v6_final.keras'),
        os.path.join(basedir, 'best_model.keras')
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                logger.info(f"Loading model: {p}")
                model = tf.keras.models.load_model(p, compile=False)
                logger.info("Model loaded successfully")
                return True
            except Exception as e1:
                logger.warning(f"Load with compile=False failed: {e1}, trying default...")
                try:
                    model = tf.keras.models.load_model(p)
                    logger.info("Model loaded successfully (default)")
                    return True
                except Exception as e2:
                    logger.error(f"Failed to load {p}: {e2}")
    logger.warning("No model could be loaded")
    return False

def preprocess_image(filepath):
    """MobileNetV3 Specific Preprocessing"""
    img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Scales pixels to range expected by MobileNetV3
    return tf.keras.applications.mobilenet_v3.preprocess_input(img_array)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- API ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(CLASS_NAMES),
        'tf_version': tf.__version__,
        'server_time': datetime.now().isoformat()
    })

@app.route('/api/warmup')
def warmup():
    """Load the model (call after deploy so first predict is fast). Blocks until done."""
    if model is not None:
        return jsonify({'model_loaded': True, 'message': 'Model already loaded'})
    with _model_load_lock:
        if model is not None:
            return jsonify({'model_loaded': True, 'message': 'Model already loaded'})
        ok = _load_model_sync()
    return jsonify({'model_loaded': ok, 'message': 'Model loaded' if ok else 'Model failed to load'}), 200 if ok else 503

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    start_time = time.time()
    try:
        # Lazy-load model in this thread on first use (avoids TensorFlow cross-thread issues)
        if model is None:
            with _model_load_lock:
                if model is None:
                    logger.info("Model not loaded; loading now (this may take 1–2 minutes)...")
                    ok = _load_model_sync()
                    if not ok:
                        return jsonify({'error': 'AI model could not be loaded on server'}), 503
            # If we didn't acquire the lock but model is now set, another request loaded it; continue
            if model is None:
                return jsonify({'error': 'AI model not loaded on server'}), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'Selected file is empty'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400

        # Save temporarily
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        try:
            # 1. Preprocess
            input_data = preprocess_image(filepath)
            
            # 2. Inference
            predictions = model(input_data, training=False)
            probabilities = predictions.numpy()[0]
            
            # 3. Post-process
            top_idx = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[top_idx]
            confidence = float(probabilities[top_idx] * 100)
            
            # Top 3
            top3_indices = probabilities.argsort()[-3:][::-1]
            top_3 = []
            for idx in top3_indices:
                top_3.append({
                    'class': CLASS_NAMES[idx],
                    'confidence': float(probabilities[idx] * 100)
                })
            
            # 4. Nutrition lookup
            portion = request.form.get('portion', 'medium')
            nutrition = get_nutrition_info(predicted_class, portion)
            
            # 5. Cleanup
            if os.path.exists(filepath): os.remove(filepath)
            
            # Build Response
            res = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'top_3': top_3,
                'timestamp': datetime.now().isoformat(),
                'inference_ms': round((time.time() - start_time) * 1000, 2)
            }
            
            if nutrition:
                res.update({
                    'nutrition': nutrition,
                    'health_indicators': get_health_indicators(nutrition),
                    'health_score': calculate_health_score(nutrition),
                    'suitability': get_dietary_suitability(nutrition)
                })
            
            logger.info(f"Prediction: {predicted_class} ({res['inference_ms']}ms)")
            return jsonify(res)

        except Exception as e:
            if os.path.exists(filepath): os.remove(filepath)
            logger.error(f"Prediction Error: {e}", exc_info=True)
            return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/classes')
def get_classes():
    return jsonify({'classes': CLASS_NAMES, 'count': len(CLASS_NAMES)})

@app.route('/api/search')
def search():
    """Text-based dish lookup by name. Returns same structure as /api/predict for displayResults."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'success': False, 'message': 'Please enter a dish name to search'}), 400

    dishes = get_all_dishes()
    query_lower = query.lower()

    # 1. Exact match (case-insensitive)
    for dish in dishes:
        if dish.lower() == query_lower:
            return _build_search_response(dish)
    # 2. Starts-with match
    for dish in dishes:
        if dish.lower().startswith(query_lower):
            return _build_search_response(dish)
    # 3. Contains match
    matches = [d for d in dishes if query_lower in d.lower()]
    if matches:
        return _build_search_response(matches[0])
    return jsonify({'success': False, 'message': f'"{query}" not found. Try another dish or upload a photo.'}), 404

def _build_search_response(predicted_class):
    nutrition = get_nutrition_info(predicted_class, portion='medium')
    if not nutrition:
        return jsonify({'success': False, 'message': 'Nutrition data unavailable for this dish'}), 404
    res = {
        'success': True,
        'predicted_class': predicted_class,
        'confidence': 100.0,
        'top_3': [{'class': predicted_class, 'confidence': 100.0}],
        'timestamp': datetime.now().isoformat(),
        'source': 'search'
    }
    res.update({
        'nutrition': nutrition,
        'health_indicators': get_health_indicators(nutrition),
        'health_score': calculate_health_score(nutrition),
        'suitability': get_dietary_suitability(nutrition)
    })
    return jsonify(res)

@app.route('/api/nutrition/<path:dish>')
def get_nutrition(dish):
    """Get nutrition info for a dish with custom portion (q=quantity, u=unit)."""
    qty = request.args.get('q') or None
    unit = request.args.get('u') or None
    nutrition = get_nutrition_info(dish, quantity=qty, unit=unit)
    if not nutrition:
        return jsonify({'success': False, 'message': f'Dish "{dish}" not found'}), 404
    return jsonify({
        'success': True,
        'nutrition': nutrition,
        'health_indicators': get_health_indicators(nutrition),
        'health_score': calculate_health_score(nutrition),
        'suitability': get_dietary_suitability(nutrition)
    })

# Auth & History Implementation
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data.get('username')).first()
    if user and user.check_password(data.get('password')):
        login_user(user)
        return jsonify({'success': True, 'username': user.username})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.json
    if User.query.filter_by(username=data.get('username')).first():
        return jsonify({'error': 'Username taken'}), 400
    user = User(username=data.get('username'))
    user.set_password(data.get('password'))
    db.session.add(user)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/auth/status')
def status():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True, 'username': current_user.username})
    return jsonify({'authenticated': False})

@app.route('/api/auth/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'success': True})

# --- STARTUP ---
def initialize():
    # Ensure all required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # CRITICAL: Create the 'instance' folder for SQLite
    instance_path = os.path.join(basedir, 'instance')
    os.makedirs(instance_path, exist_ok=True)
    
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            
    load_class_names()
    # Model loads on first /api/predict or /api/warmup (lazy) so the app can start immediately

# Run initialization once on import
initialize()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
