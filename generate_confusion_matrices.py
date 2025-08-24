#!/usr/bin/env python3
"""
Generate Confusion Matrices for K-Fold and Leave-One-Out Cross-Validation
Tests the fine-tuned model and creates visual confusion matrices.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, LeaveOneOut
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

def run_cross_validation(classifier, articles, labels, cv_type, n_folds=None):
    """Run cross-validation and return results."""
    print(f"Running {cv_type} Cross-Validation")
    print("=" * 60)
    
    if cv_type == "Leave-One-Out":
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_predictions = []
    all_true_labels = []
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(articles)):
        if cv_type == "Leave-One-Out":
            print(f"🔄 Processing Sample {fold + 1}/{len(articles)}")
        else:
            print(f"🔄 Processing Fold {fold + 1}/{n_folds}")
        
        test_articles = [articles[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        
        # Get predictions
        predictions = []
        for i, article in enumerate(test_articles):
            try:
                pred, conf, _ = classifier.classify_article(article)
                predictions.append(pred)
                
            except Exception as e:
                print(f"     ❌ Error classifying article: {e}")
                raise  # Re-raise the exception to fail fast
        
        all_predictions.extend(predictions)
        all_true_labels.extend(test_labels)
        
        # Calculate fold accuracy
        fold_accuracy = accuracy_score(test_labels, predictions)
        fold_accuracies.append(fold_accuracy)
        
        if cv_type == "Leave-One-Out":
            if fold % 10 == 0:  # Print every 10th sample
                print(f"   Sample {fold + 1} Accuracy: {fold_accuracy:.4f}")
        else:
            print(f"   Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
    
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
        'cv_type': cv_type,
        'n_folds': n_folds if n_folds else len(articles),
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

def create_confusion_matrix_plot(results, save_path):
    """Create and save a confusion matrix plot."""
    plt.figure(figsize=(10, 8))
    
    # Create confusion matrix heatmap
    cm = results['confusion_matrix']
    labels = ['Left-Leaning', 'Right-Leaning']
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {results["cv_type"]} Cross-Validation\n'
              f'Accuracy: {results["overall_accuracy"]:.4f} | '
              f'Specificity: {results["specificity"]:.4f} | '
              f'Sensitivity: {results["sensitivity"]:.4f} | '
              f'R²: {results["r2_score"]:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add text annotations
    plt.text(0.5, -0.15, f'Total Samples: {len(results["all_true_labels"])}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Confusion matrix saved to: {save_path}")

def print_results(results):
    """Print the results in a formatted way."""
    print(f"\n📊 {results['cv_type']} Results")
    print("=" * 60)
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Specificity:      {results['specificity']:.4f}")
    print(f"Sensitivity:      {results['sensitivity']:.4f}")
    print(f"R² Score:         {results['r2_score']:.4f}")
    
    if results['cv_type'] != "Leave-One-Out":
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

def create_comparison_plot(all_results):
    """Create a comparison plot of all methods."""
    methods = []
    accuracies = []
    specificities = []
    sensitivities = []
    r2_scores = []
    
    for result in all_results:
        methods.append(result['cv_type'])
        accuracies.append(result['overall_accuracy'])
        specificities.append(result['specificity'])
        sensitivities.append(result['sensitivity'])
        r2_scores.append(result['r2_score'])
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    bars1 = ax1.bar(methods, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Specificity comparison
    bars2 = ax2.bar(methods, specificities, color='lightgreen', alpha=0.7)
    ax2.set_title('Specificity Comparison')
    ax2.set_ylabel('Specificity')
    ax2.set_ylim(0, 1)
    for bar, spec in zip(bars2, specificities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{spec:.3f}', ha='center', va='bottom')
    
    # Sensitivity comparison
    bars3 = ax3.bar(methods, sensitivities, color='lightcoral', alpha=0.7)
    ax3.set_title('Sensitivity Comparison')
    ax3.set_ylabel('Sensitivity')
    ax3.set_ylim(0, 1)
    for bar, sens in zip(bars3, sensitivities):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{sens:.3f}', ha='center', va='bottom')
    
    # R² comparison
    bars4 = ax4.bar(methods, r2_scores, color='gold', alpha=0.7)
    ax4.set_title('R² Score Comparison')
    ax4.set_ylabel('R² Score')
    ax4.set_ylim(0, 1)
    for bar, r2 in zip(bars4, r2_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{r2:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cross_validation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("💾 Comparison plot saved to: cross_validation_comparison.png")

def main():
    """Main function."""
    print("🚀 Generating Confusion Matrices for Cross-Validation")
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
    
    print(f"🎯 Starting Cross-Validation with {len(articles)} articles")
    print("=" * 60)
    
    all_results = []
    
    # Run 5-fold evaluation
    results_5fold = run_cross_validation(classifier, articles, labels, "5-Fold", 5)
    print_results(results_5fold)
    create_confusion_matrix_plot(results_5fold, "confusion_matrix_5fold.png")
    all_results.append(results_5fold)
    
    print("\n" + "=" * 70)
    
    # Run 10-fold evaluation
    results_10fold = run_cross_validation(classifier, articles, labels, "10-Fold", 10)
    print_results(results_10fold)
    create_confusion_matrix_plot(results_10fold, "confusion_matrix_10fold.png")
    all_results.append(results_10fold)
    
    print("\n" + "=" * 70)
    
    # Run leave-one-out evaluation
    results_loo = run_cross_validation(classifier, articles, labels, "Leave-One-Out")
    print_results(results_loo)
    create_confusion_matrix_plot(results_loo, "confusion_matrix_leave_one_out.png")
    all_results.append(results_loo)
    
    # Create comparison plot
    create_comparison_plot(all_results)
    
    # Save all results to file
    results_summary = {}
    for result in all_results:
        cv_type = result['cv_type'].replace('-', '_').replace(' ', '_').lower()
        results_summary[cv_type] = {
            'accuracy': result['overall_accuracy'],
            'specificity': result['specificity'],
            'sensitivity': result['sensitivity'],
            'r2_score': result['r2_score'],
            'n_samples': len(result['all_true_labels'])
        }
    
    with open('confusion_matrix_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 All results saved to 'confusion_matrix_results.json'")
    print(f"📊 Generated confusion matrices:")
    print(f"   - confusion_matrix_5fold.png")
    print(f"   - confusion_matrix_10fold.png")
    print(f"   - confusion_matrix_leave_one_out.png")
    print(f"   - cross_validation_comparison.png")

if __name__ == "__main__":
    main()


