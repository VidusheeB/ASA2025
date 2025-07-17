import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import pickle
import os
import re
from collections import Counter
import PyPDF2
import io

# Set page config
st.set_page_config(
    page_title="Political Leaning Classifier",
    page_icon="🗳️",
    layout="wide"
)

class PoliticalLeaningClassifier:
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
            
            return True
            
        except FileNotFoundError:
            st.error(f"Model file not found at {self.model_path}")
            st.info("Please run 'train_model.py' first to train and save the model.")
            return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
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
                    'contribution': score * self.model.coef_[0][i]
                })
        
        # Sort by absolute contribution
        feature_analysis.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return feature_analysis

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file with enhanced capabilities"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        total_pages = len(pdf_reader.pages)
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty text
                text += page_text + "\n"
        
        # Check if we got meaningful text
        if not text.strip():
            return None, "No text could be extracted. This PDF may contain only images or scanned content."
        
        # Basic text cleaning
        text = text.strip()
        
        # Check text quality
        word_count = len(text.split())
        if word_count < 10:
            return None, f"Very little text extracted ({word_count} words). This PDF may be image-based."
        
        return text, f"Successfully extracted {word_count} words from {total_pages} pages."
        
    except Exception as e:
        return None, f"Error reading PDF: {str(e)}"

def main():
    st.title("🗳️ Political Leaning Classifier")
    st.markdown("---")
    
    # Initialize classifier
    classifier = PoliticalLeaningClassifier()
    
    # Load model
    with st.spinner("Loading trained model..."):
        if not classifier.load_model():
            st.stop()
    
        # Create tabs
    tab1, tab2 = st.tabs(["📊 Analysis Dashboard", "📝 Article Predictor"])
    
    with tab1:
        st.header("📊 Analysis Dashboard")
        
        if classifier.training_metrics and classifier.feature_names:
            # Model performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Accuracy", f"{classifier.training_metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("Features", len(classifier.feature_names))
            with col3:
                st.metric("Model Status", "✅ Loaded")
            
            # Feature importance
            st.subheader("Feature Importance")
            
            # Load feature importance from saved file
            importance_path = 'models/political_classifier_importance.csv'
            if os.path.exists(importance_path):
                importance_df = pd.read_csv(importance_path)
                
                # Top features slider
                top_n = st.slider("Number of top features to show", 5, 20, 10)
                top_features = importance_df.head(top_n)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(12, 8))
                colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
                
                ax.barh(range(len(top_features)), top_features['importance'], color=colors)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.set_xlabel('Feature Importance (Absolute Coefficient)')
                ax.set_title(f'Top {top_n} Features by Importance')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature analysis
                st.subheader("Feature Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Right-leaning indicators (positive coefficients):**")
                    right_features = importance_df[importance_df['coefficient'] > 0].head(5)
                    for _, row in right_features.iterrows():
                        st.write(f"• {row['feature']}: {row['coefficient']:.4f}")
                
                with col2:
                    st.write("**Left-leaning indicators (negative coefficients):**")
                    left_features = importance_df[importance_df['coefficient'] < 0].head(5)
                    for _, row in left_features.iterrows():
                        st.write(f"• {row['feature']}: {row['coefficient']:.4f}")
            else:
                st.warning("Feature importance file not found. Please run training again.")
    
    with tab2:
        st.header("📝 Article Predictor")
        st.write("Upload a document or paste text to predict its political leaning.")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📄 Upload PDF", "📄 Upload Text File", "✏️ Paste Text"]
        )
        
        article_text = ""
        
        if input_method == "📄 Upload PDF":
            uploaded_file = st.file_uploader(
                "Upload a PDF file",
                type=['pdf']
            )
            
            if uploaded_file is not None:
                # For news articles, standard extraction is usually best
                st.info("📰 **News Article Mode**: Standard extraction works best for articles with typed text and images.")
                
                with st.spinner("Extracting text from PDF..."):
                    article_text, extraction_status = extract_text_from_pdf(uploaded_file)
                    if article_text:
                        st.success(f"✅ PDF text extracted successfully! ({extraction_status})")
                    else:
                        st.error(f"❌ Failed to extract text from PDF: {extraction_status}")
                        st.info("💡 For image-heavy articles, consider copying the text directly or using a text converter.")
        
        elif input_method == "📄 Upload Text File":
            uploaded_file = st.file_uploader(
                "Upload a text file (.txt) or CSV file",
                type=['txt', 'csv']
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.txt'):
                        article_text = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            article_text = df['text'].iloc[0]
                        else:
                            article_text = df.iloc[0, 0]  # First column
                    st.success("✅ File loaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Paste Text
            article_text = st.text_area(
                "Paste your article text here:",
                height=200,
                placeholder="Enter the article text to analyze..."
            )
        
        if article_text:
            if st.button("🔍 Predict Political Leaning", type="primary"):
                with st.spinner("Analyzing article..."):
                    try:
                        # Make prediction
                        prediction, probability, features = classifier.predict(article_text)
                        feature_analysis = classifier.analyze_features(article_text)
                        
                        st.subheader("🎯 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if prediction == 0:
                                st.success("**Prediction: LEFT-LEANING** 🟦")
                            else:
                                st.success("**Prediction: RIGHT-LEANING** 🟥")
                        
                        with col2:
                            confidence = max(probability)
                            st.metric("Confidence", f"{confidence:.3f}")
                        
                        # Show probabilities
                        st.subheader("📊 Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Political Leaning': ['Left-leaning', 'Right-leaning'],
                            'Probability': [probability[0], probability[1]]
                        })
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = ['blue', 'red']
                        ax.bar(prob_df['Political Leaning'], prob_df['Probability'], color=colors)
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        ax.set_ylim(0, 1)
                        for i, v in enumerate(prob_df['Probability']):
                            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                        st.pyplot(fig)
                        
                        # Show top features found in the article
                        st.subheader("🔍 Key Features Detected")
                        if feature_analysis:
                            st.write("**Top features found in your article:**")
                            for i, feature in enumerate(feature_analysis[:10]):
                                leaning = "RIGHT" if feature['coefficient'] > 0 else "LEFT"
                                st.write(f"• {feature['feature']}: {feature['score']:.4f} ({leaning}-leaning)")
                        else:
                            st.info("No significant features were detected in your article.")
                        
                        # Show feature contribution analysis
                        if feature_analysis:
                            st.subheader("📈 Feature Contribution Analysis")
                            
                            # Create contribution chart
                            top_contributions = feature_analysis[:10]
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            features = [f['feature'] for f in top_contributions]
                            contributions = [f['contribution'] for f in top_contributions]
                            colors = ['red' if c > 0 else 'blue' for c in contributions]
                            
                            bars = ax.barh(range(len(features)), contributions, color=colors)
                            ax.set_yticks(range(len(features)))
                            ax.set_yticklabels(features)
                            ax.set_xlabel('Contribution to Prediction')
                            ax.set_title('Top Feature Contributions')
                            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
        
        else:
            st.info("👆 Please upload a file or paste text to analyze.")

if __name__ == "__main__":
    main() 