import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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

def train_model_and_predict(X, y):
    """Train model and get predictions"""
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
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    return y_test, y_pred, model, scaler

def create_confusion_matrix(y_true, y_pred):
    """Create and display confusion matrix"""
    print("=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    total = len(y_true)
    correct = sum(y_true == y_pred)
    
    # Calculate individual metrics
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"True Negatives (Left→Left): {tn}")
    print(f"False Positives (Left→Right): {fp}")
    print(f"False Negatives (Right→Left): {fn}")
    print(f"True Positives (Right→Right): {tp}")
    
    # Calculate rates
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for right-leaning
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for left-leaning
    precision_right = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_left = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"Sensitivity (Right-leaning recall): {sensitivity:.3f}")
    print(f"Specificity (Left-leaning recall): {specificity:.3f}")
    print(f"Precision (Right-leaning): {precision_right:.3f}")
    print(f"Precision (Left-leaning): {precision_left:.3f}")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Left-leaning', 'Right-leaning'], 
               yticklabels=['Left-leaning', 'Right-leaning'])
    
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.3f} ({correct}/{total})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add text annotations
    plt.text(0.5, -0.15, f'True Negatives: {tn} | False Positives: {fp}', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.25, f'False Negatives: {fn} | True Positives: {tp}', 
             ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrix saved to: confusion_matrix.png")
    
    return cm, accuracy

def create_detailed_confusion_matrix(y_true, y_pred):
    """Create a more detailed confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=['Left', 'Right'], 
               yticklabels=['Left', 'Right'])
    ax1.set_title('Absolute Values')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
               xticklabels=['Left', 'Right'], 
               yticklabels=['Left', 'Right'])
    ax2.set_title('Percentages (%)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed confusion matrix saved to: detailed_confusion_matrix.png")

def print_classification_report(y_true, y_pred):
    """Print detailed classification report"""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=['Left-leaning', 'Right-leaning']))

def run_confusion_matrix_analysis():
    """Run the complete confusion matrix analysis"""
    # Load data
    X, y = load_and_prepare_data()
    
    # Train model and get predictions
    y_test, y_pred, model, scaler = train_model_and_predict(X, y)
    
    # Create confusion matrix
    cm, accuracy = create_confusion_matrix(y_test, y_pred)
    
    # Create detailed confusion matrix
    create_detailed_confusion_matrix(y_test, y_pred)
    
    # Print classification report
    print_classification_report(y_test, y_pred)
    
    # Save results
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'true_labels': y_test.tolist()
    }
    
    # Save to CSV
    results_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'correct': y_test == y_pred
    })
    results_df.to_csv('confusion_matrix_results.csv', index=False)
    print(f"\nResults saved to: confusion_matrix_results.csv")
    
    return results

if __name__ == "__main__":
    run_confusion_matrix_analysis() 