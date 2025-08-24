#!/usr/bin/env python3
"""
Real Articles Evaluation for Political Leaning Classifier
Uses actual PDF documents from ASA Docs folder for comprehensive testing
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, classification_report
from article_classifier import ArticlePoliticalClassifier
from document_processor import DocumentProcessor
import os
from dotenv import load_dotenv
from datetime import datetime
import glob

load_dotenv()

def load_real_articles():
    """Load real articles from PDF files in ASA Docs folder."""
    articles = []
    labels = []
    file_paths = []
    
    # Path to the documents folder
    docs_path = "ASA Docs"
    
    # Get all PDF files
    pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        
        # Determine label from filename
        if "Left.pdf" in filename:
            label = "left-leaning"
        elif "Right.pdf" in filename:
            label = "right-leaning"
        else:
            continue  # Skip files that don't match the pattern
        
        try:
            # Extract text from PDF directly
            with open(pdf_file, 'rb') as f:
                file_content = f.read()
                extracted_text, success = doc_processor.extract_from_pdf(file_content)
                
                if success and extracted_text:
                    # Clean and truncate text if too long (API limits)
                    cleaned_text = extracted_text.strip()
                    if len(cleaned_text) > 8000:  # Truncate if too long
                        cleaned_text = cleaned_text[:8000] + "..."
                    
                    articles.append(cleaned_text)
                    labels.append(label)
                    file_paths.append(filename)
                    print(f"✅ Loaded {filename} ({label}) - {len(cleaned_text)} chars")
                else:
                    print(f"❌ Failed to extract text from {filename}")
                    
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    
    return articles, labels, file_paths

def run_cross_validation(classifier, articles, labels, file_paths, n_folds):
    """Run cross-validation and return detailed results."""
    print(f"\n{'='*60}")
    print(f"Running {n_folds}-Fold Cross-Validation on Real Articles")
    print(f"{'='*60}")
    
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_predictions = []
    all_true_labels = []
    all_file_paths = []
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(articles)):
        print(f"Processing fold {fold + 1}/{n_folds}")
        
        test_articles = [articles[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        test_files = [file_paths[i] for i in test_idx]
        
        # Get predictions
        predictions = []
        confidences = []
        for i, (article, file_path) in enumerate(zip(test_articles, test_files)):
            try:
                pred, conf, _ = classifier.classify_article(article)
                predictions.append(pred)
                confidences.append(conf)
                print(f"  {file_path}: {pred} (confidence: {conf:.2f})")
            except Exception as e:
                print(f"  Error in classification for {file_path}: {e}")
                predictions.append("left-leaning")  # Default fallback
                confidences.append(0.5)
        
        all_predictions.extend(predictions)
        all_true_labels.extend(test_labels)
        all_file_paths.extend(test_files)
        
        # Calculate fold-specific metrics
        fold_accuracy = accuracy_score(test_labels, predictions)
        fold_precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
        fold_recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
        fold_f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
        
        # Calculate specificity and sensitivity for this fold
        conf_matrix = confusion_matrix(test_labels, predictions)
        if conf_matrix.shape == (2, 2):
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            specificity = 0
            sensitivity = 0
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': fold_accuracy,
            'precision': fold_precision,
            'recall': fold_recall,
            'f1_score': fold_f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'true_labels': test_labels,
            'confidences': confidences,
            'file_paths': test_files
        })
        
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    overall_precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    overall_recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    
    # Calculate overall specificity and sensitivity
    overall_conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    if overall_conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = overall_conf_matrix.ravel()
        overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        overall_specificity = 0
        overall_sensitivity = 0
    
    # Calculate R²
    y_true_numeric = [1 if label == 'right-leaning' else 0 for label in all_true_labels]
    y_pred_numeric = [1 if pred == 'right-leaning' else 0 for pred in all_predictions]
    try:
        r2 = r2_score(y_true_numeric, y_pred_numeric)
    except:
        r2 = 0.0
    
    # Display results
    print(f"\n{n_folds}-Fold Overall Results:")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1-Score: {overall_f1:.4f}")
    print(f"Specificity: {overall_specificity:.4f}")
    print(f"Sensitivity: {overall_sensitivity:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(overall_conf_matrix)
    
    return {
        'n_folds': n_folds,
        'overall_accuracy': overall_accuracy,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1,
        'overall_specificity': overall_specificity,
        'overall_sensitivity': overall_sensitivity,
        'r2_score': r2,
        'overall_confusion_matrix': overall_conf_matrix,
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels,
        'all_file_paths': all_file_paths,
        'fold_results': fold_results
    }

def generate_confusion_matrix_plot(conf_matrix, title, filename):
    """Generate and save confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['left-leaning', 'right-leaning'],
                yticklabels=['left-leaning', 'right-leaning'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(results_5fold, results_10fold, model_name):
    """Save detailed results to CSV and text files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comprehensive results DataFrame
    results_data = []
    
    # Add 5-fold results
    results_data.append({
        'Test_Type': '5-Fold CV',
        'Accuracy': results_5fold['overall_accuracy'],
        'Precision': results_5fold['overall_precision'],
        'Recall': results_5fold['overall_recall'],
        'F1_Score': results_5fold['overall_f1_score'],
        'Specificity': results_5fold['overall_specificity'],
        'Sensitivity': results_5fold['overall_sensitivity'],
        'R2_Score': results_5fold['r2_score']
    })
    
    # Add 10-fold results
    results_data.append({
        'Test_Type': '10-Fold CV',
        'Accuracy': results_10fold['overall_accuracy'],
        'Precision': results_10fold['overall_precision'],
        'Recall': results_10fold['overall_recall'],
        'F1_Score': results_10fold['overall_f1_score'],
        'Specificity': results_10fold['overall_specificity'],
        'Sensitivity': results_10fold['overall_sensitivity'],
        'R2_Score': results_10fold['r2_score']
    })
    
    # Save to CSV
    results_df = pd.DataFrame(results_data)
    csv_filename = f"real_articles_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    
    # Save detailed text report
    report_filename = f"real_articles_report_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write("REAL ARTICLES POLITICAL LEANING CLASSIFIER EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Used: {model_name}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("5-FOLD CROSS-VALIDATION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {results_5fold['overall_accuracy']:.4f} ({results_5fold['overall_accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results_5fold['overall_precision']:.4f}\n")
        f.write(f"Recall: {results_5fold['overall_recall']:.4f}\n")
        f.write(f"F1-Score: {results_5fold['overall_f1_score']:.4f}\n")
        f.write(f"Specificity: {results_5fold['overall_specificity']:.4f}\n")
        f.write(f"Sensitivity: {results_5fold['overall_sensitivity']:.4f}\n")
        f.write(f"R² Score: {results_5fold['r2_score']:.4f}\n\n")
        
        f.write("10-FOLD CROSS-VALIDATION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {results_10fold['overall_accuracy']:.4f} ({results_10fold['overall_accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results_10fold['overall_precision']:.4f}\n")
        f.write(f"Recall: {results_10fold['overall_recall']:.4f}\n")
        f.write(f"F1-Score: {results_10fold['overall_f1_score']:.4f}\n")
        f.write(f"Specificity: {results_10fold['overall_specificity']:.4f}\n")
        f.write(f"Sensitivity: {results_10fold['overall_sensitivity']:.4f}\n")
        f.write(f"R² Score: {results_10fold['r2_score']:.4f}\n\n")
        
        f.write("CONFUSION MATRICES\n")
        f.write("-" * 40 + "\n")
        f.write("5-Fold Confusion Matrix:\n")
        f.write(str(results_5fold['overall_confusion_matrix']) + "\n\n")
        f.write("10-Fold Confusion Matrix:\n")
        f.write(str(results_10fold['overall_confusion_matrix']) + "\n\n")
        
        f.write("DETAILED PREDICTIONS (5-Fold)\n")
        f.write("-" * 40 + "\n")
        for i, (true_label, pred, file_path) in enumerate(zip(results_5fold['all_true_labels'], 
                                                             results_5fold['all_predictions'], 
                                                             results_5fold['all_file_paths'])):
            status = "✓" if true_label == pred else "✗"
            f.write(f"{status} {file_path}: True={true_label}, Predicted={pred}\n")
        
        f.write("\nDETAILED PREDICTIONS (10-Fold)\n")
        f.write("-" * 40 + "\n")
        for i, (true_label, pred, file_path) in enumerate(zip(results_10fold['all_true_labels'], 
                                                             results_10fold['all_predictions'], 
                                                             results_10fold['all_file_paths'])):
            status = "✓" if true_label == pred else "✗"
            f.write(f"{status} {file_path}: True={true_label}, Predicted={pred}\n")
    
    return csv_filename, report_filename

def main():
    """Main function."""
    print("🚀 Real Articles Political Leaning Classifier Evaluation")
    print("=" * 70)
    
    # Initialize classifier
    try:
        classifier = ArticlePoliticalClassifier()
        model_name = classifier.model_name
        print(f"✅ Classifier initialized successfully")
        print(f"📋 Model: {model_name}")
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        return
    
    # Load real articles
    print("\n📚 Loading real articles from PDF files...")
    articles, labels, file_paths = load_real_articles()
    
    if not articles:
        print("❌ No articles loaded. Please check the ASA Docs folder.")
        return
    
    print(f"✅ Loaded {len(articles)} articles")
    print(f"   Left-leaning: {labels.count('left-leaning')}")
    print(f"   Right-leaning: {labels.count('right-leaning')}")
    
    # Run 5-fold cross-validation
    results_5fold = run_cross_validation(classifier, articles, labels, file_paths, 5)
    
    # Run 10-fold cross-validation
    results_10fold = run_cross_validation(classifier, articles, labels, file_paths, 10)
    
    # Generate confusion matrix plots
    print("\n📊 Generating confusion matrix plots...")
    generate_confusion_matrix_plot(
        results_5fold['overall_confusion_matrix'], 
        "5-Fold Cross-Validation Confusion Matrix (Real Articles)",
        "real_articles_confusion_matrix_5fold.png"
    )
    generate_confusion_matrix_plot(
        results_10fold['overall_confusion_matrix'], 
        "10-Fold Cross-Validation Confusion Matrix (Real Articles)",
        "real_articles_confusion_matrix_10fold.png"
    )
    
    # Save detailed results
    print("\n💾 Saving detailed results...")
    csv_file, report_file = save_detailed_results(results_5fold, results_10fold, model_name)
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    
    print(f"{'Metric':<15} {'5-Fold':<12} {'10-Fold':<12} {'Difference':<12}")
    print("-" * 55)
    print(f"{'Accuracy':<15} {results_5fold['overall_accuracy']:<12.4f} {results_10fold['overall_accuracy']:<12.4f} {results_10fold['overall_accuracy'] - results_5fold['overall_accuracy']:<12.4f}")
    print(f"{'Precision':<15} {results_5fold['overall_precision']:<12.4f} {results_10fold['overall_precision']:<12.4f} {results_10fold['overall_precision'] - results_5fold['overall_precision']:<12.4f}")
    print(f"{'Recall':<15} {results_5fold['overall_recall']:<12.4f} {results_10fold['overall_recall']:<12.4f} {results_10fold['overall_recall'] - results_5fold['overall_recall']:<12.4f}")
    print(f"{'F1-Score':<15} {results_5fold['overall_f1_score']:<12.4f} {results_10fold['overall_f1_score']:<12.4f} {results_10fold['overall_f1_score'] - results_5fold['overall_f1_score']:<12.4f}")
    print(f"{'Specificity':<15} {results_5fold['overall_specificity']:<12.4f} {results_10fold['overall_specificity']:<12.4f} {results_10fold['overall_specificity'] - results_5fold['overall_specificity']:<12.4f}")
    print(f"{'Sensitivity':<15} {results_5fold['overall_sensitivity']:<12.4f} {results_10fold['overall_sensitivity']:<12.4f} {results_10fold['overall_sensitivity'] - results_5fold['overall_sensitivity']:<12.4f}")
    print(f"{'R² Score':<15} {results_5fold['r2_score']:<12.4f} {results_10fold['r2_score']:<12.4f} {results_10fold['r2_score'] - results_5fold['r2_score']:<12.4f}")
    
    print(f"\n{'='*70}")
    print("FILES GENERATED")
    print(f"{'='*70}")
    print(f"📊 Confusion Matrix (5-fold): real_articles_confusion_matrix_5fold.png")
    print(f"📊 Confusion Matrix (10-fold): real_articles_confusion_matrix_10fold.png")
    print(f"📋 Detailed Results: {csv_file}")
    print(f"📄 Comprehensive Report: {report_file}")
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
