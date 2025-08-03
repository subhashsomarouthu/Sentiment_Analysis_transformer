# Individual Model Docker Image - Transformer with Word2Vec
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and source code
COPY models/ ./models/
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src

# Set environment variables for this specific model
ENV MODEL_NAME="Transformer_Word2Vec_Model"
ENV MODEL_ID="model_transformer_w2v"
ENV MODEL_PORT="5000"
ENV MODEL_VERSION="1.0"

# Expose port (will be mapped to different ports in final container)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Create a non-root user for security
RUN useradd -m -u 1000 modeluser && chown -R modeluser:modeluser /app
USER modeluser

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--preload", "src.app:app"]