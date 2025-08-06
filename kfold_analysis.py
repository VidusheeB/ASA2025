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

def perform_kfold_analysis(X, y, k_values=[3, 5, 10]):
    """Perform k-fold cross-validation analysis"""
    print("=" * 60)
    print("K-FOLD CROSS-VALIDATION ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    for k in k_values:
        print(f"\n--- K = {k} ---")
        
        # Initialize k-fold
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        
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
        results[k] = {
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
        print(f"Individual fold scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Print classification report
        print(f"\nClassification Report (K={k}):")
        print(classification_report(y, cv_predictions, target_names=['Left-leaning', 'Right-leaning']))
    
    return results

def create_confusion_matrices(X, y, results):
    """Create confusion matrices for each k value"""
    print("\n" + "=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (k, result) in enumerate(results.items()):
        cm = confusion_matrix(y, result['predictions'])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Left', 'Right'], 
                   yticklabels=['Left', 'Right'],
                   ax=axes[i])
        axes[i].set_title(f'K = {k} (Accuracy: {result["accuracy"]:.3f})')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('kfold_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrices saved to: kfold_confusion_matrices.png")

def create_accuracy_comparison(results):
    """Create accuracy comparison plot"""
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)
    
    k_values = list(results.keys())
    accuracies = [results[k]['accuracy'] for k in k_values]
    accuracies_std = [results[k]['accuracy_std'] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, accuracies, yerr=accuracies_std, marker='o', capsize=5)
    plt.xlabel('K (Number of Folds)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('K-Fold Cross-Validation Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add accuracy labels
    for i, (k, acc, std) in enumerate(zip(k_values, accuracies, accuracies_std)):
        plt.annotate(f'{acc:.3f}±{std:.3f}', 
                    (k, acc), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('kfold_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Accuracy comparison saved to: kfold_accuracy_comparison.png")

def save_detailed_results(results):
    """Save detailed results to CSV"""
    detailed_results = []
    
    for k, result in results.items():
        # Add fold-by-fold results
        for fold_idx, score in enumerate(result['cv_scores']):
            detailed_results.append({
                'k': k,
                'fold': fold_idx + 1,
                'accuracy': score,
                'correct': result['correct'],
                'left_to_right': result['left_to_right'],
                'right_to_left': result['right_to_left'],
                'total': result['total'],
                'correct_proportion': result['correct_proportion'],
                'left_to_right_proportion': result['left_to_right_proportion'],
                'right_to_left_proportion': result['right_to_left_proportion']
            })
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv('kfold_analysis_results.csv', index=False)
    print("Detailed results saved to: kfold_analysis_results.csv")

def run_kfold_analysis():
    """Run the complete k-fold analysis"""
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Perform k-fold analysis
    results = perform_kfold_analysis(X, y, k_values=[3, 5, 10])
    
    # Create visualizations
    create_confusion_matrices(X, y, results)
    create_accuracy_comparison(results)
    
    # Save detailed results
    save_detailed_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for k, result in results.items():
        print(f"K={k}: {result['accuracy']:.3f} (±{result['accuracy_std']:.3f})")
        print(f"  Left→Right: {result['left_to_right_proportion']:.2%}")
        print(f"  Right→Left: {result['right_to_left_proportion']:.2%}")
    
    return results

if __name__ == "__main__":
    run_kfold_analysis() 