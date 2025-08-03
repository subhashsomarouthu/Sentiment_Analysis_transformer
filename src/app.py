import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
import json
import logging
from inference import SentimentPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get model info from environment variables (for container orchestration)
MODEL_NAME = os.getenv('MODEL_NAME', 'Transformer_Word2Vec_Model')
MODEL_ID = os.getenv('MODEL_ID', 'model_transformer_w2v')
MODEL_PORT = os.getenv('MODEL_PORT', '5000')
MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0')

# Initialize predictor (load model once when app starts)
try:
    predictor = SentimentPredictor()
    logger.info(f"Model {MODEL_NAME} loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model {MODEL_NAME}: {e}")
    predictor = None

@app.route('/', methods=['GET'])
def home():
    """Home page with API information"""
    return jsonify({
        'message': f'Welcome to {MODEL_NAME} API',
        'model_name': MODEL_NAME,
        'model_id': MODEL_ID,
        'architecture': 'Non-causal Transformer with Word2Vec (200D), 8 heads, dual pooling',
        'version': MODEL_VERSION,
        'port': MODEL_PORT,
        'status': 'ready' if predictor else 'error',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict',
            'predict_batch': 'POST /predict_batch', 
            'model_info': 'GET /model_info'
        },
        'example_usage': {
            'single_prediction': {
                'url': '/predict',
                'method': 'POST',
                'body': {'text': 'I love this product!'}
            },
            'batch_prediction': {
                'url': '/predict_batch', 
                'method': 'POST',
                'body': {'texts': ['I love this!', 'This is terrible.']}
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy', 'message': 'Model is ready'})

@app.route('/predict', methods=['POST'])
def predict():
    """Single text prediction endpoint"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field in request'}), 400
        
        text = data['text']
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        result = predictor.predict_single(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch text prediction endpoint"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field in request'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        # Validate each text
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                return jsonify({'error': f'Text at index {i} must be a non-empty string'}), 400
        
        # Limit batch size
        if len(texts) > 100:
            return jsonify({'error': 'Batch size cannot exceed 100 texts'}), 400
        
        results = predictor.predict_batch(texts)
        return jsonify({'predictions': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_params': predictor.params,
        'vocab_size': len(predictor.tokenizer.word_index),
        'classes': list(predictor.reverse_label_map.values()),
        'input_shape': predictor.model.input_shape,
        'output_shape': predictor.model.output_shape
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False)
    