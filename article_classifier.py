import os
import json
import openai
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv
from text_processor import PoliticalTextProcessor

load_dotenv()

class ArticlePoliticalClassifier:
    """
    A class to classify political leaning of articles using a fine-tuned ChatGPT model.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the classifier with the fine-tuned model.
        
        Args:
            model_name: The name of the fine-tuned model (e.g., 'ft:gpt-4.1-2025-04-14:...')
            api_key: OpenAI API key (will use environment variable if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        openai.api_key = self.api_key
        
        # Fine-tuned model name - required
        self.model_name = model_name or os.getenv('FINE_TUNED_MODEL_NAME')
        if not self.model_name:
            raise ValueError("Fine-tuned model name is required. Set FINE_TUNED_MODEL_NAME environment variable or pass model_name parameter.")
        
        # Initialize text processor
        self.text_processor = PoliticalTextProcessor()
    
    def classify_article(self, article_text: str) -> Tuple[str, float, Dict]:
        """
        Classify political leaning of an article based on its text content.
        
        Args:
            article_text: The article text to classify
            
        Returns:
            Tuple of (prediction, confidence_score, analysis_data)
        """
        # Create the message format for article text classification
        messages = [
            {
                "role": "system",
                "content": "You are a political leaning classifier. Analyze the article text and predict whether it is left-leaning or right-leaning."
            },
            {
                "role": "user",
                "content": f"Article text: {article_text}"
            }
        ]
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=50
            )
            
            prediction_text = response.choices[0].message.content.strip()
            
            # Display the full response for debugging
            print(f"📝 Full Model Response: {prediction_text}")
            
            # Extract prediction and confidence from detailed response
            prediction_text_lower = prediction_text.lower()
            
            # Look for explicit left/right-leaning mentions
            if "left-leaning" in prediction_text_lower or "left leaning" in prediction_text_lower:
                prediction = "left-leaning"
                confidence = 0.9
            elif "right-leaning" in prediction_text_lower or "right leaning" in prediction_text_lower:
                prediction = "right-leaning"
                confidence = 0.9
            # Look for political indicators in the analysis
            elif any(word in prediction_text_lower for word in ["liberal", "progressive", "democrat", "biden", "left"]):
                prediction = "left-leaning"
                confidence = 0.8
            elif any(word in prediction_text_lower for word in ["conservative", "republican", "trump", "right", "america first"]):
                prediction = "right-leaning"
                confidence = 0.8
            # Look for neutral indicators that might suggest right-leaning (criticizing Biden)
            elif "neutral" in prediction_text_lower and any(word in prediction_text_lower for word in ["criticizing", "biden", "administration"]):
                prediction = "right-leaning"
                confidence = 0.7
            # Handle cases where model says it can't determine - default to left-leaning for now
            elif "unable to determine" in prediction_text_lower or "neutral" in prediction_text_lower:
                prediction = "left-leaning"  # Default fallback
                confidence = 0.5
            else:
                # If we can't determine, default to left-leaning
                prediction = "left-leaning"
                confidence = 0.5
            
            print(f"🎯 Extracted Prediction: {prediction} (confidence: {confidence})")
            
            # Add analysis data
            analysis_data = {
                'prediction': prediction,
                'confidence': confidence,
                'article_length': len(article_text),
                'model_used': self.model_name
            }
            
            return prediction, confidence, analysis_data
            
        except Exception as e:
            raise Exception(f"Error calling fine-tuned model: {e}")
    
    def batch_classify_articles(self, articles: List[str]) -> List[Tuple[str, float, Dict]]:
        """
        Classify multiple articles at once.
        
        Args:
            articles: List of article texts
            
        Returns:
            List of (prediction, confidence, analysis_data) tuples
        """
        results = []
        for article_text in articles:
            result = self.classify_article(article_text)
            results.append(result)
        return results
    
    def evaluate_model_on_articles(self, articles: List[str], true_labels: List[str]) -> Dict:
        """
        Evaluate the model performance on article texts.
        
        Args:
            articles: List of article texts
            true_labels: List of true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        confidences = []
        analysis_data_list = []
        
        for article_text in articles:
            pred, conf, analysis = self.classify_article(article_text)
            predictions.append(pred)
            confidences.append(conf)
            analysis_data_list.append(analysis)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Filter out error predictions
        valid_indices = [i for i, pred in enumerate(predictions) if pred != "error"]
        if not valid_indices:
            return {"error": "No valid predictions made"}
        
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]
        
        accuracy = accuracy_score(valid_true_labels, valid_predictions)
        report = classification_report(valid_true_labels, valid_predictions, output_dict=True)
        conf_matrix = confusion_matrix(valid_true_labels, valid_predictions)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "predictions": predictions,
            "confidences": confidences,
            "analysis_data": analysis_data_list,
            "valid_predictions": valid_predictions,
            "valid_true_labels": valid_true_labels
        } 