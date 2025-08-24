#!/usr/bin/env python3
"""
K-Fold Cross-Validation with Real Articles
Tests the fine-tuned model on all 50 real PDF articles using K-fold cross-validation.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from article_classifier import ArticlePoliticalClassifier
from document_processor import DocumentProcessor

def get_article_files():
    """Get all PDF files from the ASA Docs directory."""
    base_dir = "ASA Docs"
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory '{base_dir}' not found!")
        return [], []
    
    left_articles = []
    right_articles = []
    
    # Look for PDF files in the directory
    for filename in os.listdir(base_dir):
        if filename.endswith('.pdf'):
            filepath = os.path.join(base_dir, filename)
            
            # Determine leaning based on filename
            if 'Left' in filename:
                left_articles.append(filepath)
            elif 'Right' in filename:
                right_articles.append(filepath)
            else:
                print(f"⚠️  Warning: {filename} doesn't contain 'Left' or 'Right' in name")
    
    return left_articles, right_articles

def load_article_text(filepath, document_processor):
    """Load and extract text from a PDF file."""
    try:
        with open(filepath, 'rb') as f:
            file_content = f.read()
        
        article_text, success = document_processor.extract_from_pdf(file_content)
        
        if not success:
            raise ValueError(f"Failed to extract text from {os.path.basename(filepath)}")
        
        if not article_text or len(article_text.strip()) < 50:
            raise ValueError(f"{os.path.basename(filepath)} has insufficient text")
        
        return article_text
        
    except Exception as e:
        print(f"❌ Error loading {os.path.basename(filepath)}: {e}")
        raise

def run_kfold_evaluation(classifier, articles, labels, n_folds):
    """Run K-fold cross-validation."""
    print(f"Running {n_folds}-Fold Cross-Validation")
    print("=" * 60)
    
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_predictions = []
    all_true_labels = []
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(articles)):
        print(f"🔄 Processing Fold {fold + 1}/{n_folds}")
        print(f"   Test set size: {len(test_idx)} articles")
        
        test_articles = [articles[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        
        # Get predictions
        predictions = []
        for i, article in enumerate(test_articles):
            try:
                pred, conf, _ = classifier.classify_article(article)
                predictions.append(pred)
                
                # Show some predictions for first fold
                if fold == 0 and i < 3:
                    status = "✅" if pred == test_labels[i] else "❌"
                    print(f"     {status} {os.path.basename(test_articles[i])}: {pred} (expected: {test_labels[i]})")
                
            except Exception as e:
                print(f"     ❌ Error classifying article: {e}")
                raise  # Re-raise the exception to fail fast
        
        all_predictions.extend(predictions)
        all_true_labels.extend(test_labels)
        
        # Calculate fold accuracy
        fold_accuracy = accuracy_score(test_labels, predictions)
        fold_accuracies.append(fold_accuracy)
        print(f"   Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
        print()
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    conf_matrix = confusion_matrix(all_true_labels, all_predictions, labels=['left-leaning', 'right-leaning'])
    
    # Calculate specificity and sensitivity
    tn, fp, fn, tp = conf_matrix.ravel()
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
        'n_folds': n_folds,
        'overall_accuracy': overall_accuracy,
        'fold_accuracies': fold_accuracies,
        'confusion_matrix': conf_matrix,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'r2_score': r2_score_value,
        'classification_report': classification_report(all_true_labels, all_predictions),
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels
    }
    
    return results

def print_results(results, fold_type):
    """Print the results in a formatted way."""
    print(f"\n📊 {fold_type} Results")
    print("=" * 60)
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Specificity:      {results['specificity']:.4f}")
    print(f"Sensitivity:      {results['sensitivity']:.4f}")
    print(f"R² Score:         {results['r2_score']:.4f}")
    
    print(f"\nFold Accuracies:")
    for i, acc in enumerate(results['fold_accuracies']):
        print(f"  Fold {i+1}: {acc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Left    Right")
    print(f"Actual Left     {results['confusion_matrix'][0][0]:6d}  {results['confusion_matrix'][0][1]:6d}")
    print(f"      Right     {results['confusion_matrix'][1][0]:6d}  {results['confusion_matrix'][1][1]:6d}")
    
    print(f"\nDetailed Classification Report:")
    print(results['classification_report'])

def main():
    """Main function."""
    print("🚀 K-Fold Cross-Validation with Real Articles")
    print("=" * 70)
    
    # Initialize classifier and document processor
    classifier = ArticlePoliticalClassifier()
    document_processor = DocumentProcessor()
    print("✅ Classifier initialized successfully")
    
    # Get article files
    left_articles, right_articles = get_article_files()
    
    if not left_articles or not right_articles:
        print("❌ No articles found!")
        return
    
    print(f"📖 Loading all articles...")
    
    # Load all articles
    articles = []
    labels = []
    
    # Load left-leaning articles
    for pdf_path in left_articles:
        try:
            article_text = load_article_text(pdf_path, document_processor)
            articles.append(article_text)
            labels.append("left-leaning")
            print(f"  ✅ Loaded: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"  ❌ Failed to load {os.path.basename(pdf_path)}: {e}")
    
    # Load right-leaning articles
    for pdf_path in right_articles:
        try:
            article_text = load_article_text(pdf_path, document_processor)
            articles.append(article_text)
            labels.append("right-leaning")
            print(f"  ✅ Loaded: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"  ❌ Failed to load {os.path.basename(pdf_path)}: {e}")
    
    print(f"📊 Loaded {len(articles)} articles successfully")
    print(f"   Left-leaning: {labels.count('left-leaning')}")
    print(f"   Right-leaning: {labels.count('right-leaning')}")
    
    if len(articles) == 0:
        print("❌ No articles could be loaded!")
        return
    
    print(f"🎯 Starting K-Fold Evaluation with {len(articles)} articles")
    print("=" * 60)
    
    # Run 5-fold evaluation
    results_5fold = run_kfold_evaluation(classifier, articles, labels, 5)
    print_results(results_5fold, "5-Fold Cross-Validation")
    
    print("\n" + "=" * 70)
    
    # Run 10-fold evaluation
    results_10fold = run_kfold_evaluation(classifier, articles, labels, 10)
    print_results(results_10fold, "10-Fold Cross-Validation")
    
    # Save results to file
    results_summary = {
        '5_fold': {
            'accuracy': results_5fold['overall_accuracy'],
            'specificity': results_5fold['specificity'],
            'sensitivity': results_5fold['sensitivity'],
            'r2_score': results_5fold['r2_score']
        },
        '10_fold': {
            'accuracy': results_10fold['overall_accuracy'],
            'specificity': results_10fold['specificity'],
            'sensitivity': results_10fold['sensitivity'],
            'r2_score': results_10fold['r2_score']
        }
    }
    
    with open('kfold_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to 'kfold_results.json'")

if __name__ == "__main__":
    main()



