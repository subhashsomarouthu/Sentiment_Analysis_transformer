import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import json

# Custom PositionalEmbedding layer (must match your training code exactly)
class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim, max_len, embedding_matrix=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        # Handle embedding matrix (for deployment compatibility)
        if embedding_matrix is not None:
            self.embedding_matrix = np.array(embedding_matrix)
        else:
            # Create placeholder - actual weights will be loaded from saved model
            self.embedding_matrix = np.zeros((vocab_size, embedding_dim))

        self.token_embed = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[self.embedding_matrix] if embedding_matrix is not None else None,
            trainable=False
        )
        self.pos_embed = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embedding_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_embed(positions)
        x = self.token_embed(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_len': self.max_len,
            # Don't save embedding_matrix to avoid conflicts
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Remove embedding_matrix from config if it exists to avoid conflicts
        config = config.copy()
        config.pop('embedding_matrix', None)
        return cls(**config)

class SentimentPredictor:
    def __init__(self, model_path='models/transformer_sentiment_model.keras', 
                 tokenizer_path='models/tokenizer.pkl',
                 label_map_path='models/label_map.pkl',
                 params_path='models/model_params.pkl'):
        
        # Load parameters
        with open(params_path, 'rb') as f:
            self.params = pickle.load(f)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load label mapping
        with open(label_map_path, 'rb') as f:
            self.label_map = pickle.load(f)
        
        # Create reverse label mapping
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load model with custom objects
        self.model = load_model(model_path, custom_objects={'PositionalEmbedding': PositionalEmbedding})
        
        print("Model loaded successfully!")
        print(f"Model expects input shape: {self.model.input_shape}")
        print(f"Model output classes: {list(self.reverse_label_map.values())}")
    
    def preprocess_text(self, text):
        """Preprocess a single text input"""
        if isinstance(text, str):
            text = [text]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(text)
        
        # Pad sequences
        padded = pad_sequences(sequences, 
                             maxlen=self.params['maxlen'], 
                             padding='post', 
                             truncating='post')
        
        return padded
    
    def predict_single(self, text):
        """Predict sentiment for a single text"""
        processed = self.preprocess_text(text)
        prediction = self.model.predict(processed, verbose=0)
        
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        sentiment = self.reverse_label_map[predicted_class]
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'all_scores': {
                self.reverse_label_map[i]: float(score) 
                for i, score in enumerate(prediction[0])
            }
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        processed = self.preprocess_text(texts)
        predictions = self.model.predict(processed, verbose=0)
        
        results = []
        for i, text in enumerate(texts):
            predicted_class = np.argmax(predictions[i])
            confidence = float(predictions[i][predicted_class])
            sentiment = self.reverse_label_map[predicted_class]
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'all_scores': {
                    self.reverse_label_map[j]: float(score) 
                    for j, score in enumerate(predictions[i])
                }
            })
        
        return results

# Test function
def test_model():
    try:
        predictor = SentimentPredictor()
        
        # Test with sample texts
        test_texts = [
            "I love this product! It's amazing and works perfectly!",
            "This is terrible, I hate it. Worst purchase ever.",
            "It's okay, nothing special. Average quality."
        ]
        
        print("\n=== Testing Model ===")
        for text in test_texts:
            result = predictor.predict_single(text)
            print(f"Text: {result['text']}")
            print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
            print(f"All scores: {result['all_scores']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all model files are in the same directory!")

if __name__ == "__main__":
    test_model()