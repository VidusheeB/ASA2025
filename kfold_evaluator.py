import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from article_classifier import ArticlePoliticalClassifier

class KFoldEvaluator:
    """
    A class to perform K-fold cross-validation on the fine-tuned political classifier.
    """
    
    def __init__(self, classifier: ArticlePoliticalClassifier, k_folds: int = 5):
        """
        Initialize the K-fold evaluator.
        
        Args:
            classifier: The fine-tuned classifier instance
            k_folds: Number of folds for cross-validation
        """
        self.classifier = classifier
        self.k_folds = k_folds
        self.results = []
        
    def load_training_data(self, jsonl_file: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load training data from JSONL file and extract TF-IDF scores and labels.
        
        Args:
            jsonl_file: Path to the JSONL training file
            
        Returns:
            Tuple of (tfidf_dataframe, labels_list)
        """
        tfidf_data = []
        labels = []
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Extract TF-IDF scores from the user message
                user_content = data['messages'][1]['content']
                tfidf_start = user_content.find('{')
                tfidf_end = user_content.rfind('}') + 1
                tfidf_json = user_content[tfidf_start:tfidf_end]
                tfidf_scores = json.loads(tfidf_json)
                
                # Extract label from assistant message
                assistant_content = data['messages'][2]['content']
                if "left-leaning" in assistant_content.lower():
                    label = "left-leaning"
                elif "right-leaning" in assistant_content.lower():
                    label = "right-leaning"
                else:
                    continue  # Skip unclear labels
                
                tfidf_data.append(tfidf_scores)
                labels.append(label)
        
        # Convert to DataFrame
        df = pd.DataFrame(tfidf_data)
        return df, labels
    
    def _create_articles_from_tfidf(self, tfidf_data: pd.DataFrame) -> List[str]:
        """
        Create synthetic article texts from TF-IDF scores for evaluation.
        
        Args:
            tfidf_data: DataFrame with TF-IDF scores
            
        Returns:
            List of synthetic article texts
        """
        articles = []
        
        for _, row in tfidf_data.iterrows():
            # Create a synthetic article based on TF-IDF scores
            # Terms with higher scores will appear more frequently
            article_parts = []
            
            for term, score in row.items():
                if score > 0:
                    # Repeat the term based on its TF-IDF score
                    repetitions = max(1, int(score * 10))  # Scale the score
                    article_parts.extend([term] * repetitions)
            
            if article_parts:
                # Create a coherent article by joining terms
                article_text = " ".join(article_parts)
                # Add some context to make it more realistic
                article_text = f"Article discussing {article_text}. This content addresses various political issues."
            else:
                # If no significant terms, create a neutral article
                article_text = "This is a neutral article with minimal political content."
            
            articles.append(article_text)
        
        return articles
    
    def perform_kfold_evaluation(self, tfidf_data: pd.DataFrame, labels: List[str]) -> Dict:
        """
        Perform K-fold cross-validation.
        
        Args:
            tfidf_data: DataFrame with TF-IDF scores
            labels: List of true labels
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(tfidf_data)):
            print(f"Processing fold {fold + 1}/{self.k_folds}")
            
            # Get test data for this fold
            test_data = tfidf_data.iloc[test_idx]
            test_labels = [labels[i] for i in test_idx]
            
            # Convert TF-IDF data back to article texts for evaluation
            # Since we're working with the training data that has TF-IDF scores,
            # we'll create synthetic article texts based on the scores
            test_articles = self._create_articles_from_tfidf(test_data)
            
            # Evaluate on test data
            fold_eval = self.classifier.evaluate_model_on_articles(test_articles, test_labels)
            
            if "error" in fold_eval:
                print(f"Error in fold {fold + 1}: {fold_eval['error']}")
                continue
            
            fold_results.append({
                'fold': fold + 1,
                'accuracy': fold_eval['accuracy'],
                'predictions': fold_eval['predictions'],
                'true_labels': test_labels,
                'confusion_matrix': fold_eval['confusion_matrix'],
                'classification_report': fold_eval['classification_report']
            })
            
            all_predictions.extend(fold_eval['valid_predictions'])
            all_true_labels.extend(fold_eval['valid_true_labels'])
        
        # Calculate overall metrics
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        overall_precision = precision_score(all_true_labels, all_predictions, average='weighted')
        overall_recall = recall_score(all_true_labels, all_predictions, average='weighted')
        overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
        
        # Overall confusion matrix
        overall_conf_matrix = confusion_matrix(all_true_labels, all_predictions)
        
        # Calculate specificity and sensitivity
        tn, fp, fn, tp = overall_conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate R² score
        from sklearn.metrics import r2_score
        y_true_numeric = [1 if label == 'right-leaning' else 0 for label in all_true_labels]
        y_pred_numeric = [1 if pred == 'right-leaning' else 0 for pred in all_predictions]
        try:
            r2_score_value = r2_score(y_true_numeric, y_pred_numeric)
        except:
            r2_score_value = 0.0
        
        results = {
            'fold_results': fold_results,
            'overall_metrics': {
                'accuracy': overall_accuracy,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'r2_score': r2_score_value
            },
            'overall_confusion_matrix': overall_conf_matrix,
            'all_predictions': all_predictions,
            'all_true_labels': all_true_labels
        }
        
        self.results = results
        return results
    
    def generate_confusion_matrix_plot(self, save_path: str = "kfold_confusion_matrix.png"):
        """
        Generate and save confusion matrix visualization.
        
        Args:
            save_path: Path to save the confusion matrix plot
        """
        if not self.results:
            print("No results available. Run evaluation first.")
            return
        
        conf_matrix = self.results['overall_confusion_matrix']
        labels = ['left-leaning', 'right-leaning']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('K-Fold Cross-Validation Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def generate_fold_performance_plot(self, save_path: str = "kfold_performance.png"):
        """
        Generate fold-wise performance comparison plot.
        
        Args:
            save_path: Path to save the performance plot
        """
        if not self.results:
            print("No results available. Run evaluation first.")
            return
        
        fold_accuracies = [fold['accuracy'] for fold in self.results['fold_results']]
        fold_numbers = [fold['fold'] for fold in self.results['fold_results']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(fold_numbers, fold_accuracies, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=self.results['overall_metrics']['accuracy'], color='r', 
                   linestyle='--', label=f'Overall Accuracy: {self.results["overall_metrics"]["accuracy"]:.3f}')
        plt.xlabel('Fold Number')
        plt.ylabel('Accuracy')
        plt.title('K-Fold Cross-Validation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance plot saved to {save_path}")
    
    def save_results_to_csv(self, save_path: str = "kfold_results.csv"):
        """
        Save detailed results to CSV file.
        
        Args:
            save_path: Path to save the CSV results
        """
        if not self.results:
            print("No results available. Run evaluation first.")
            return
        
        # Create detailed results DataFrame
        results_data = []
        for fold_result in self.results['fold_results']:
            fold_data = {
                'fold': fold_result['fold'],
                'accuracy': fold_result['accuracy'],
                'precision_left': fold_result['classification_report']['left-leaning']['precision'],
                'recall_left': fold_result['classification_report']['left-leaning']['recall'],
                'f1_left': fold_result['classification_report']['left-leaning']['f1-score'],
                'precision_right': fold_result['classification_report']['right-leaning']['precision'],
                'recall_right': fold_result['classification_report']['right-leaning']['recall'],
                'f1_right': fold_result['classification_report']['right-leaning']['f1-score']
            }
            results_data.append(fold_data)
        
        # Add overall metrics
        overall_data = {
            'fold': 'Overall',
            'accuracy': self.results['overall_metrics']['accuracy'],
            'precision': self.results['overall_metrics']['precision'],
            'recall': self.results['overall_metrics']['recall'],
            'f1_score': self.results['overall_metrics']['f1_score'],
            'specificity': self.results['overall_metrics']['specificity'],
            'sensitivity': self.results['overall_metrics']['sensitivity']
        }
        results_data.append(overall_data)
        
        df = pd.DataFrame(results_data)
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
    
    def print_summary(self):
        """
        Print a comprehensive summary of the evaluation results.
        """
        if not self.results:
            print("No results available. Run evaluation first.")
            return
        
        print("\n" + "="*60)
        print("K-FOLD CROSS-VALIDATION RESULTS")
        print("="*60)
        
        # Overall metrics
        metrics = self.results['overall_metrics']
        print(f"\nOverall Performance Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  R² Score:    {metrics['r2_score']:.4f}")
        
        # Fold-wise results
        print(f"\nFold-wise Results:")
        for fold_result in self.results['fold_results']:
            print(f"  Fold {fold_result['fold']}: Accuracy = {fold_result['accuracy']:.4f}")
        
        # Confusion matrix
        conf_matrix = self.results['overall_confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Left    Right")
        print(f"Actual Left     {conf_matrix[0][0]:6d}  {conf_matrix[0][1]:6d}")
        print(f"      Right     {conf_matrix[1][0]:6d}  {conf_matrix[1][1]:6d}")
        
        print("\n" + "="*60) 