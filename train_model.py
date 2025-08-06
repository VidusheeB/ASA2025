import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent pop-ups
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, data_path='Scores_100.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_metrics = {}
        
    def load_data(self):
        """Load and prepare the data"""
        print(f"Loading data from {self.data_path}")
        data = pd.read_csv(self.data_path)
        
        # Separate features and target
        y = data['Label']
        X = data.drop('Label', axis=1)
        self.feature_names = X.columns.tolist()
        
        print(f"Dataset shape: {data.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Class distribution:")
        print(y.value_counts())
        
        return X, y
    
    def split_data(self, X, y, test_size=0.25, random_state=42):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the logistic regression model"""
        print("Training logistic regression model...")
        
        # Store training data for cross-validation
        self.X_train = X_train
        self.y_train = y_train
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        print("Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions with optimal threshold
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        y_pred = (y_pred_proba[:, 1] >= 0.2).astype(int)  # Use optimal threshold of 0.2
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate cross-validation scores
        cv_folds = min(3, len(self.X_train) // 2)  # Adjust for small dataset
        X_train_scaled = self.scaler.transform(self.X_train)
        cv_accuracy = cross_val_score(self.model, X_train_scaled, self.y_train, 
                                     cv=cv_folds, scoring='accuracy')
        cv_r2 = cross_val_score(self.model, X_train_scaled, self.y_train, 
                               cv=cv_folds, scoring='r2')
        
        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Cross-validation Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
        print(f"Cross-validation R²: {cv_r2.mean():.4f} (+/- {cv_r2.std() * 2:.4f})")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                 target_names=['Left-leaning', 'Right-leaning']))
        
        # Store metrics
        self.training_metrics = {
            'accuracy': accuracy,
            'cv_accuracy': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_r2': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return self.training_metrics
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            return None
        
        feature_importance = np.abs(self.model.coef_[0])
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance,
            'coefficient': self.model.coef_[0]
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def save_model(self, model_path='models/political_classifier.pkl'):
        """Save the trained model and related components"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
        
        # Save feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            importance_path = model_path.replace('.pkl', '_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"Feature importance saved to {importance_path}")
    
    def plot_results(self):
        """Plot training results"""
        if not self.training_metrics:
            print("No training metrics available for plotting")
            return
        
        # Confusion Matrix
        cm = self.training_metrics['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Left-leaning', 'Right-leaning'],
                   yticklabels=['Left-leaning', 'Right-leaning'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature Importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            top_features = importance_df.head(15)
            
            plt.figure(figsize=(12, 8))
            colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
            
            plt.barh(range(len(top_features)), top_features['importance'], color=colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance (Absolute Coefficient)')
            plt.title('Top 15 Features by Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function to train and save the model"""
    print("=" * 60)
    print("POLITICAL LEANING CLASSIFIER - MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer('Scores.csv')
    
    # Load data
    X, y = trainer.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train model
    trainer.train_model(X_train, y_train)
    
    # Evaluate model
    metrics = trainer.evaluate_model(X_test, y_test)
    
    # Get feature importance
    importance_df = trainer.get_feature_importance()
    if importance_df is not None:
        print("\n" + "=" * 50)
        print("FEATURE IMPORTANCE")
        print("=" * 50)
        print("Top 10 most important features:")
        print(importance_df.head(10)[['feature', 'importance', 'coefficient']])
    
    # Save model
    trainer.save_model()
    
    # Plot results (saves plots to files instead of showing pop-ups)
    trainer.plot_results()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("Model and related files saved to 'models/' directory")
    print("Plots saved as:")
    print("  - models/confusion_matrix.png")
    print("  - models/feature_importance.png")
    print("You can now use 'predict.py' to make predictions on new articles")

if __name__ == "__main__":
    main() 