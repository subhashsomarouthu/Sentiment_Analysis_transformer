# Sentiment Analysis Transformer

A Docker-deployable sentiment analysis API using a non-causal Transformer model with pretrained Word2Vec embeddings and positional encoding.

## Model Architecture
- **Base**: Non-causal Transformer with multi-head attention
- **Embeddings**: Pretrained Word2Vec (Google News 300d) with positional encoding
- **Classes**: 3 sentiment classes
- **Max Sequence Length**: 150 tokens
- **Vocabulary Size**: 10,000 words

## Project Structure
```
sentiment-transformer-app/
├── models/
│   ├── transformer_sentiment_model.keras
│   ├── tokenizer.pkl
│   ├── label_map.pkl
│   └── model_params.pkl
├── src/
│   ├── inference.py
│   └── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Quick Start

### 1. Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Test model loading
cd src && python inference.py

# Run Flask app
python app.py
```

### 2. Docker Deployment
```bash
# Build image
docker build -t sentiment-transformer .

# Run container
docker run -p 5000:5000 sentiment-transformer

# Test health
curl http://localhost:5000/health
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
    "text": "I love this product!"
}
```

**Response:**
```json
{
    "text": "I love this product!",
    "sentiment": "positive",
    "confidence": 0.95,
    "all_scores": {
        "positive": 0.95,
        "negative": 0.03,
        "neutral": 0.02
    }
}
```

### Batch Prediction
```bash
POST /predict_batch
Content-Type: application/json

{
    "texts": [
        "I love this!",
        "This is terrible.",
        "It's okay."
    ]
}
```

### Model Info
```bash
GET /model_info
```

## Model Requirements
- Python 3.11
- TensorFlow 2.18.0
- Pre-trained Word2Vec embeddings (embedded in model)
- 4 model files in `/models/` directory

## Environment Variables
- `PORT`: Server port (default: 5000)
- `WORKERS`: Gunicorn workers (default: 1)

## Performance Notes
- Model uses frozen Word2Vec embeddings (no training required)
- Optimized for CPU inference
- Supports batch processing (max 100 texts per request)
- Memory efficient with pre-loaded model

## Troubleshooting
- Ensure all 4 model files are present in `/models/`
- Check TensorFlow compatibility with your system
- For GPU support, modify Dockerfile to include CUDA libraries
