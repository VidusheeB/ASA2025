import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_set2_25w_model():
    """Train model on Set 2 with 25 words and find optimal threshold"""
    print("Training Set 2 25-words model...")
    
    # Load the Set 2 25 words dataset
    data_path = 'data/ASA2025_ TF-IDF - Set 2 - Ready 25.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return None, None
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Handle target column (first column contains target values)
    y = data.iloc[:, 0]  # First column as target
    X = data.drop(data.columns[0], axis=1)  # Drop first column
    
    print(f"Target distribution: {y.value_counts()}")
    print(f"Features shape: {X.shape}")
    
    # Clean data - remove rows with NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"After cleaning: {X.shape[0]} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions
    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold using training data
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    print("\nFinding optimal threshold...")
    for threshold in thresholds:
        train_pred = (train_proba >= threshold).astype(int)
        f1 = f1_score(y_train, train_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f} (F1-score: {best_f1:.3f})")
    
    # Evaluate with optimal threshold
    test_pred = (test_proba >= best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    
    # Calculate sensitivity and specificity
    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nTest Set Performance (threshold: {best_threshold:.3f}):")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall/Sensitivity: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Specificity: {specificity:.3f}")
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Left-leaning', 'Right-leaning'],
               yticklabels=['Left-leaning', 'Right-leaning'])
    plt.title(f'Set 2 25-words Model\nAcc: {accuracy:.3f}, Sens: {sensitivity:.3f}, Spec: {specificity:.3f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/set2_25w_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'optimal_threshold': best_threshold,
        'performance': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    }
    
    with open('models/set2_25w_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: models/set2_25w_model.pkl")
    print(f"Confusion matrix saved to: models/set2_25w_confusion_matrix.png")
    
    return model_data, best_threshold

if __name__ == "__main__":
    model_data, optimal_threshold = train_set2_25w_model()
    
    if model_data:
        print(f"\n✅ Model training complete!")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"Test accuracy: {model_data['performance']['accuracy']:.3f}")
    else:
        print("❌ Model training failed!")
