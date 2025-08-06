import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load the dataset and prepare for analysis"""
    print("Loading dataset...")
    data = pd.read_csv('Scores.csv')
    
    # Separate features and target
    y = data['Label']
    X = data.drop('Label', axis=1)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Left-leaning articles: {sum(y == 0)}")
    print(f"Right-leaning articles: {sum(y == 1)}")
    
    return X, y

def train_model_and_get_probabilities(X, y):
    """Train model and get probability predictions"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Get probability predictions
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of right-leaning
    
    return y_test, y_proba, model, scaler

def find_optimal_threshold(y_true, y_proba):
    """Find optimal threshold by analyzing different thresholds"""
    print("=" * 60)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # Define threshold range
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    results = []
    
    for threshold in thresholds:
        # Make predictions based on threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix elements
        tn, fp, fn, tp = np.bincount(y_true * 2 + y_pred, minlength=4)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        })
    
    # Find optimal thresholds for different metrics
    results_df = pd.DataFrame(results)
    
    optimal_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    optimal_precision = results_df.loc[results_df['precision'].idxmax()]
    optimal_recall = results_df.loc[results_df['recall'].idxmax()]
    optimal_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    
    print(f"Optimal threshold for accuracy: {optimal_accuracy['threshold']:.3f} (Accuracy: {optimal_accuracy['accuracy']:.3f})")
    print(f"Optimal threshold for precision: {optimal_precision['threshold']:.3f} (Precision: {optimal_precision['precision']:.3f})")
    print(f"Optimal threshold for recall: {optimal_recall['threshold']:.3f} (Recall: {optimal_recall['recall']:.3f})")
    print(f"Optimal threshold for F1-score: {optimal_f1['threshold']:.3f} (F1: {optimal_f1['f1_score']:.3f})")
    
    return results_df, optimal_accuracy, optimal_precision, optimal_recall, optimal_f1

def create_threshold_analysis_plots(results_df):
    """Create plots for threshold analysis"""
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS PLOTS")
    print("=" * 60)
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Metrics vs Threshold
    ax1.plot(results_df['threshold'], results_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['precision'], 'r-', label='Precision', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['f1_score'], 'm-', label='F1-Score', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confusion Matrix Elements vs Threshold
    ax2.plot(results_df['threshold'], results_df['true_positives'], 'g-', label='True Positives', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['false_positives'], 'r-', label='False Positives', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['true_negatives'], 'b-', label='True Negatives', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['false_negatives'], 'orange', label='False Negatives', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Count')
    ax2.set_title('Confusion Matrix Elements vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    
    ax4.plot(recall, precision, 'r-', linewidth=2, label='Precision-Recall Curve')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Threshold analysis plots saved to: threshold_analysis_plots.png")

def create_optimal_threshold_comparison(results_df, optimal_accuracy, optimal_precision, optimal_recall, optimal_f1):
    """Create comparison of different optimal thresholds"""
    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLD COMPARISON")
    print("=" * 60)
    
    # Create predictions for each optimal threshold
    thresholds_to_test = [
        ('Default (0.5)', 0.5),
        ('Optimal Accuracy', optimal_accuracy['threshold']),
        ('Optimal Precision', optimal_precision['threshold']),
        ('Optimal Recall', optimal_recall['threshold']),
        ('Optimal F1', optimal_f1['threshold'])
    ]
    
    comparison_results = []
    
    for name, threshold in thresholds_to_test:
        y_pred = (y_proba >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = np.bincount(y_true * 2 + y_pred, minlength=4)
        
        comparison_results.append({
            'threshold_name': name,
            'threshold_value': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        })
        
        print(f"\n{name} (Threshold: {threshold:.3f}):")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Create comparison plot
    comparison_df = pd.DataFrame(comparison_results)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    ax1.bar(comparison_df['threshold_name'], comparison_df['accuracy'], color='skyblue')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Precision comparison
    ax2.bar(comparison_df['threshold_name'], comparison_df['precision'], color='lightcoral')
    ax2.set_title('Precision Comparison')
    ax2.set_ylabel('Precision')
    ax2.tick_params(axis='x', rotation=45)
    
    # Recall comparison
    ax3.bar(comparison_df['threshold_name'], comparison_df['recall'], color='lightgreen')
    ax3.set_title('Recall Comparison')
    ax3.set_ylabel('Recall')
    ax3.tick_params(axis='x', rotation=45)
    
    # F1-Score comparison
    ax4.bar(comparison_df['threshold_name'], comparison_df['f1_score'], color='gold')
    ax4.set_title('F1-Score Comparison')
    ax4.set_ylabel('F1-Score')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('optimal_threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Optimal threshold comparison saved to: optimal_threshold_comparison.png")
    
    return comparison_df

def save_detailed_results(results_df, comparison_df):
    """Save detailed results to CSV"""
    results_df.to_csv('threshold_analysis_results.csv', index=False)
    comparison_df.to_csv('optimal_threshold_comparison.csv', index=False)
    print("\nDetailed results saved to:")
    print("- threshold_analysis_results.csv")
    print("- optimal_threshold_comparison.csv")

def run_optimal_threshold_analysis():
    """Run the complete optimal threshold analysis"""
    # Load data
    X, y = load_and_prepare_data()
    
    # Train model and get probabilities
    global y_true, y_proba
    y_true, y_proba, model, scaler = train_model_and_get_probabilities(X, y)
    
    # Find optimal thresholds
    results_df, optimal_accuracy, optimal_precision, optimal_recall, optimal_f1 = find_optimal_threshold(y_true, y_proba)
    
    # Create plots
    create_threshold_analysis_plots(results_df)
    
    # Create comparison
    comparison_df = create_optimal_threshold_comparison(results_df, optimal_accuracy, optimal_precision, optimal_recall, optimal_f1)
    
    # Save results
    save_detailed_results(results_df, comparison_df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLD SUMMARY")
    print("=" * 60)
    print(f"Best F1-Score threshold: {optimal_f1['threshold']:.3f}")
    print(f"Best Accuracy threshold: {optimal_accuracy['threshold']:.3f}")
    print(f"Best Precision threshold: {optimal_precision['threshold']:.3f}")
    print(f"Best Recall threshold: {optimal_recall['threshold']:.3f}")
    
    return results_df, comparison_df

if __name__ == "__main__":
    run_optimal_threshold_analysis() 