import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
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

def perform_k5_analysis(X, y):
    """Perform k=5 fold cross-validation analysis"""
    print("=" * 60)
    print("K=5 FOLD CROSS-VALIDATION ANALYSIS")
    print("=" * 60)
    
    # Initialize k=5 fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize model
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')
    cv_predictions = cross_val_predict(model, X_scaled, y, cv=kfold)
    
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
    
    # Store results
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
        'right_to_left_proportion': right_to_left_proportion
    }
    
    print(f"Accuracy: {accuracy:.3f} (±{accuracy_std:.3f})")
    print(f"Correct: {correct}/{total} ({correct_proportion:.2%})")
    print(f"Left→Right: {left_to_right} ({left_to_right_proportion:.2%})")
    print(f"Right→Left: {right_to_left} ({right_to_left_proportion:.2%})")
    
    # Print individual fold scores
    print(f"\nIndividual fold scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.3f}")
    
    # Print classification report
    print(f"\nClassification Report (K=5):")
    print(classification_report(y, cv_predictions, target_names=['Left-leaning', 'Right-leaning']))
    
    return results

def create_confusion_matrix(X, y, results):
    """Create confusion matrix for k=5"""
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (K=5)")
    print("=" * 60)
    
    cm = confusion_matrix(y, results['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Left-leaning', 'Right-leaning'], 
               yticklabels=['Left-leaning', 'Right-leaning'])
    plt.title(f'K=5 Cross-Validation Confusion Matrix\nAccuracy: {results["accuracy"]:.3f} (±{results["accuracy_std"]:.3f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    plt.text(0.5, -0.15, f'True Negatives: {tn} | False Positives: {fp}', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.25, f'False Negatives: {fn} | True Positives: {tp}', 
             ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('k5_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to: k5_confusion_matrix.png")

def create_fold_performance_plot(results):
    """Create fold performance visualization"""
    print("\n" + "=" * 60)
    print("FOLD PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    fold_scores = results['cv_scores']
    
    plt.figure(figsize=(12, 6))
    
    # Bar chart of fold scores
    plt.subplot(1, 2, 1)
    folds = range(1, len(fold_scores) + 1)
    bars = plt.bar(folds, fold_scores, color=['#2E8B57' if score >= 0.5 else '#DC143C' for score in fold_scores])
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Individual Fold Performance')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, fold_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    # Add mean line
    plt.axhline(y=results['accuracy'], color='red', linestyle='--', label=f'Mean: {results["accuracy"]:.3f}')
    plt.legend()
    
    # Box plot of scores
    plt.subplot(1, 2, 2)
    plt.boxplot(fold_scores, labels=['K=5 Folds'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy Distribution')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('k5_fold_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fold performance saved to: k5_fold_performance.png")

def analyze_fold_details(X, y, results):
    """Analyze details of each fold"""
    print("\n" + "=" * 60)
    print("DETAILED FOLD ANALYSIS")
    print("=" * 60)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        correct = sum(y_test == y_pred)
        left_to_right = sum((y_test == 0) & (y_pred == 1))
        right_to_left = sum((y_test == 1) & (y_pred == 0))
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"  Accuracy: {accuracy:.3f} ({correct}/{len(y_test)})")
        print(f"  Left→Right: {left_to_right}, Right→Left: {right_to_left}")

def save_detailed_results(results):
    """Save detailed results to CSV"""
    detailed_results = []
    
    # Add fold-by-fold results
    for fold_idx, score in enumerate(results['cv_scores'], 1):
        detailed_results.append({
            'fold': fold_idx,
            'accuracy': score,
            'correct': results['correct'],
            'left_to_right': results['left_to_right'],
            'right_to_left': results['right_to_left'],
            'total': results['total'],
            'correct_proportion': results['correct_proportion'],
            'left_to_right_proportion': results['left_to_right_proportion'],
            'right_to_left_proportion': results['right_to_left_proportion']
        })
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv('k5_analysis_results.csv', index=False)
    print("\nDetailed results saved to: k5_analysis_results.csv")

def run_k5_analysis():
    """Run the complete k=5 analysis"""
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Perform k=5 analysis
    results = perform_k5_analysis(X, y)
    
    # Create visualizations
    create_confusion_matrix(X, y, results)
    create_fold_performance_plot(results)
    
    # Analyze fold details
    analyze_fold_details(X, y, results)
    
    # Save detailed results
    save_detailed_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("K=5 SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {results['accuracy']:.3f} (±{results['accuracy_std']:.3f})")
    print(f"Correct Predictions: {results['correct']}/{results['total']} ({results['correct_proportion']:.2%})")
    print(f"Left→Right Errors: {results['left_to_right']} ({results['left_to_right_proportion']:.2%})")
    print(f"Right→Left Errors: {results['right_to_left']} ({results['right_to_left_proportion']:.2%})")
    
    return results

if __name__ == "__main__":
    run_k5_analysis() 