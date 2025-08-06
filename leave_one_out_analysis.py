import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    feature_names = X.columns.tolist()
    
    print(f"Dataset shape: {data.shape}")
    print(f"Left-leaning articles: {sum(y == 0)}")
    print(f"Right-leaning articles: {sum(y == 1)}")
    
    return X, y, feature_names

def perform_leave_one_out_analysis(X, y):
    """Perform leave-one-out cross-validation analysis"""
    print("=" * 60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION ANALYSIS")
    print("=" * 60)
    
    # Initialize leave-one-out
    loo = LeaveOneOut()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize model
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=loo, scoring='accuracy')
    cv_predictions = cross_val_predict(model, X_scaled, y, cv=loo)
    
    # Calculate metrics
    accuracy = np.mean(cv_scores)
    accuracy_std = np.std(cv_scores)
    
    # Analyze prediction patterns
    correct = sum(y == cv_predictions)
    left_to_right = sum((y == 0) & (cv_predictions == 1))
    right_to_left = sum((y == 1) & (cv_predictions == 0))
    total = len(y)
    
    # Calculate proportions
    correct_proportion = correct / total
    left_to_right_proportion = left_to_right / total
    right_to_left_proportion = right_to_left / total
    
    # Store detailed results
    detailed_results = []
    for i, (true_label, pred_label, score) in enumerate(zip(y, cv_predictions, cv_scores)):
        detailed_results.append({
            'sample_id': i + 1,
            'true_label': true_label,
            'predicted_label': pred_label,
            'correct': true_label == pred_label,
            'fold_accuracy': score,
            'true_class': 'Left-leaning' if true_label == 0 else 'Right-leaning',
            'predicted_class': 'Left-leaning' if pred_label == 0 else 'Right-leaning'
        })
    
    results = {
        'accuracy': accuracy,
        'accuracy_std': accuracy_std,
        'cv_scores': cv_scores,
        'predictions': cv_predictions,
        'correct': correct,
        'left_to_right': left_to_right,
        'right_to_left': right_to_left,
        'total': total,
        'correct_proportion': correct_proportion,
        'left_to_right_proportion': left_to_right_proportion,
        'right_to_left_proportion': right_to_left_proportion,
        'detailed_results': detailed_results
    }
    
    print(f"Accuracy: {accuracy:.3f} (±{accuracy_std:.3f})")
    print(f"Correct: {correct}/{total} ({correct_proportion:.2%})")
    print(f"Left→Right: {left_to_right} ({left_to_right_proportion:.2%})")
    print(f"Right→Left: {right_to_left} ({right_to_left_proportion:.2%})")
    
    # Print individual sample results
    print(f"\nIndividual sample results:")
    for result in detailed_results:
        status = "✓" if result['correct'] else "✗"
        print(f"  Sample {result['sample_id']:2d}: {result['true_class']:12s} → {result['predicted_class']:12s} {status}")
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y, cv_predictions, target_names=['Left-leaning', 'Right-leaning']))
    
    return results

def create_confusion_matrix(X, y, results):
    """Create confusion matrix"""
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    
    cm = confusion_matrix(y, results['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Left', 'Right'], 
               yticklabels=['Left', 'Right'])
    plt.title(f'Leave-One-Out Confusion Matrix (Accuracy: {results["accuracy"]:.3f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('leave_one_out_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to: leave_one_out_confusion_matrix.png")

def create_accuracy_distribution(results):
    """Create accuracy distribution plot"""
    print("\n" + "=" * 60)
    print("ACCURACY DISTRIBUTION")
    print("=" * 60)
    
    # Count correct vs incorrect predictions
    correct_count = sum(1 for r in results['detailed_results'] if r['correct'])
    incorrect_count = len(results['detailed_results']) - correct_count
    
    labels = ['Correct', 'Incorrect']
    sizes = [correct_count, incorrect_count]
    colors = ['#2E8B57', '#DC143C']
    
    plt.figure(figsize=(10, 6))
    
    # Pie chart
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Prediction Distribution')
    
    # Bar chart
    plt.subplot(1, 2, 2)
    plt.bar(labels, sizes, color=colors)
    plt.title('Prediction Counts')
    plt.ylabel('Number of Predictions')
    
    for i, v in enumerate(sizes):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('leave_one_out_accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Accuracy distribution saved to: leave_one_out_accuracy_distribution.png")

def analyze_misclassifications(results):
    """Analyze misclassification patterns"""
    print("\n" + "=" * 60)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    misclassified = [r for r in results['detailed_results'] if not r['correct']]
    
    print(f"Total misclassifications: {len(misclassified)}")
    
    left_to_right = [r for r in misclassified if r['true_class'] == 'Left-leaning' and r['predicted_class'] == 'Right-leaning']
    right_to_left = [r for r in misclassified if r['true_class'] == 'Right-leaning' and r['predicted_class'] == 'Left-leaning']
    
    print(f"Left→Right misclassifications: {len(left_to_right)}")
    print(f"Right→Left misclassifications: {len(right_to_left)}")
    
    if left_to_right:
        print(f"\nLeft→Right misclassified samples: {[r['sample_id'] for r in left_to_right]}")
    
    if right_to_left:
        print(f"Right→Left misclassified samples: {[r['sample_id'] for r in right_to_left]}")

def save_detailed_results(results):
    """Save detailed results to CSV"""
    results_df = pd.DataFrame(results['detailed_results'])
    results_df.to_csv('leave_one_out_results.csv', index=False)
    print("\nDetailed results saved to: leave_one_out_results.csv")

def run_leave_one_out_analysis():
    """Run the complete leave-one-out analysis"""
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Perform leave-one-out analysis
    results = perform_leave_one_out_analysis(X, y)
    
    # Create visualizations
    create_confusion_matrix(X, y, results)
    create_accuracy_distribution(results)
    
    # Analyze misclassifications
    analyze_misclassifications(results)
    
    # Save detailed results
    save_detailed_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Leave-One-Out Accuracy: {results['accuracy']:.3f} (±{results['accuracy_std']:.3f})")
    print(f"Correct Predictions: {results['correct']}/{results['total']} ({results['correct_proportion']:.2%})")
    print(f"Left→Right Errors: {results['left_to_right']} ({results['left_to_right_proportion']:.2%})")
    print(f"Right→Left Errors: {results['right_to_left']} ({results['right_to_left_proportion']:.2%})")
    
    return results

if __name__ == "__main__":
    run_leave_one_out_analysis() 