import pandas as pd
import numpy as np
import pickle
import re
from collections import Counter
import argparse
import sys

class PoliticalLeaningPredictor:
    def __init__(self, model_path='models/political_classifier.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_metrics = None
        
    def load_model(self):
        """Load the trained model and related components"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.training_metrics = model_data.get('training_metrics', {})
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Number of features: {len(self.feature_names)}")
            
            if self.training_metrics:
                print(f"Training accuracy: {self.training_metrics.get('accuracy', 'N/A'):.4f}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            print("Please run 'train_model.py' first to train and save the model.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_features(self, text):
        """Extract TF-IDF features from text based on vocabulary"""
        if not self.feature_names:
            raise ValueError("Feature names not loaded. Please load the model first.")
        
        # Simple word counting approach
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        
        # Create feature vector based on vocabulary
        features = []
        for term in self.feature_names:
            # Check for exact matches and partial matches
            count = 0
            for word in word_counts:
                if term.lower() in word.lower() or word.lower() in term.lower():
                    count += word_counts[word]
            features.append(count)
        
        # Normalize by text length
        if len(words) > 0:
            features = [f / len(words) for f in features]
        
        return features
    
    def predict(self, text):
        """Predict political leaning for given text"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # Extract features
        features = self.extract_features(text)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        if hasattr(self.model, 'predict') and hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
        else:
            raise ValueError("Model does not have required predict methods")
        
        return prediction, probability, features
    
    def analyze_features(self, text):
        """Analyze which features were found in the text"""
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        features = self.extract_features(text)
        
        # Create feature analysis
        feature_analysis = []
        for i, (feature, score) in enumerate(zip(self.feature_names, features)):
            if score > 0 and hasattr(self.model, 'coef_'):
                feature_analysis.append({
                    'feature': feature,
                    'score': score,
                    'coefficient': self.model.coef_[0][i],
                    'contribution': score * self.model.coef_[0][i],
                    'leaning': 'RIGHT' if self.model.coef_[0][i] > 0 else 'LEFT'
                })
        
        # Sort by absolute contribution
        feature_analysis.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return feature_analysis
    
    def get_prediction_explanation(self, text):
        """Get detailed explanation of prediction"""
        prediction, probability, features = self.predict(text)
        feature_analysis = self.analyze_features(text)
        
        # Calculate total contributions
        right_contributions = sum([f['contribution'] for f in feature_analysis if f['contribution'] > 0])
        left_contributions = sum([f['contribution'] for f in feature_analysis if f['contribution'] < 0])
        
        explanation = {
            'prediction': prediction,
            'confidence': max(probability),
            'probabilities': probability,
            'right_contributions': right_contributions,
            'left_contributions': abs(left_contributions),
            'total_contribution': right_contributions + left_contributions,
            'feature_analysis': feature_analysis
        }
        
        return explanation

def predict_from_text(text, model_path='models/political_classifier.pkl'):
    """Convenience function to predict from text"""
    predictor = PoliticalLeaningPredictor(model_path)
    
    if not predictor.load_model():
        return None, None, None, None
    
    try:
        prediction, probability, features = predictor.predict(text)
        feature_analysis = predictor.analyze_features(text)
        
        return prediction, probability, features, feature_analysis
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None, None

def main():
    """Main function for command-line prediction"""
    parser = argparse.ArgumentParser(description='Predict political leaning of text')
    parser.add_argument('--text', '-t', type=str, help='Text to analyze')
    parser.add_argument('--file', '-f', type=str, help='File containing text to analyze')
    parser.add_argument('--model', '-m', type=str, default='models/political_classifier.pkl',
                       help='Path to trained model')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PoliticalLeaningPredictor(args.model)
    
    if not predictor.load_model():
        sys.exit(1)
    
    # Get text to analyze
    text = None
    
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    elif args.interactive:
        print("Enter your article text (press Ctrl+D when finished):")
        text = sys.stdin.read()
    else:
        print("Please provide text using --text, --file, or --interactive")
        parser.print_help()
        sys.exit(1)
    
    if not text:
        print("No text provided")
        sys.exit(1)
    
    # Make prediction
    try:
        prediction, probability, features = predictor.predict(text)
        feature_analysis = predictor.analyze_features(text)
        
        print("\n" + "=" * 60)
        print("POLITICAL LEANING PREDICTION")
        print("=" * 60)
        
        # Prediction result
        if prediction == 0:
            print("🎯 Prediction: LEFT-LEANING 🟦")
        else:
            print("🎯 Prediction: RIGHT-LEANING 🟥")
        
        print(f"Confidence: {max(probability):.3f}")
        
        # Probabilities
        print(f"\nProbabilities:")
        print(f"  Left-leaning: {probability[0]:.3f}")
        print(f"  Right-leaning: {probability[1]:.3f}")
        
        # Detailed analysis
        explanation = predictor.get_prediction_explanation(text)
        
        print(f"\n📊 PREDICTION ANALYSIS:")
        print(f"Right-leaning contributions: {explanation['right_contributions']:.4f}")
        print(f"Left-leaning contributions: {explanation['left_contributions']:.4f}")
        print(f"Total contribution strength: {explanation['total_contribution']:.4f}")
        
        # Feature analysis
        if feature_analysis:
            print(f"\n🔍 Top features found in your text:")
            for i, feature in enumerate(feature_analysis[:10]):
                print(f"  {i+1}. {feature['feature']}: {feature['score']:.4f} ({feature['leaning']}-leaning, contribution: {feature['contribution']:.4f})")
        
        # Text preview
        print(f"\nText preview: {text[:100]}...")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 