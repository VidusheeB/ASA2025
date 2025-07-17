import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PoliticalLeaningAnalyzer:
    def __init__(self, data_path='Scores.csv'):
        """
        Initialize the analyzer with the scores data
        
        Args:
            data_path: Path to the CSV file with labels and TF-IDF scores
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """
        Load and prepare the data
        """
        print("Loading data from", self.data_path)
        self.data = pd.read_csv(self.data_path)
        
        # Separate features and target
        self.y = self.data['Label']
        self.X = self.data.drop('Label', axis=1)
        self.feature_names = self.X.columns.tolist()
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Class distribution:")
        print(self.y.value_counts())
        print(f"Left-leaning (0): {sum(self.y == 0)}")
        print(f"Right-leaning (1): {sum(self.y == 1)}")
        
        return self
    
    def split_data(self, test_size=0.25, random_state=42):
        """
        Split data into training and testing sets
        """
        # Adjust test_size for small datasets
        if self.data is not None and len(self.data) < 10:
            test_size = 0.25  # Use 25% for very small datasets
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        return self
    
    def train_model(self, tune_hyperparameters=False):
        """
        Train logistic regression model
        """
        if tune_hyperparameters and self.X_train is not None and len(self.X_train) >= 6:
            print("Tuning hyperparameters...")
            # Simplified parameter grid for small datasets
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],  # Only L2 for small datasets
                'class_weight': [None, 'balanced']
            }
            
            # Use fewer CV folds for small datasets
            cv_folds = min(3, len(self.X_train) // 2)
            
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
                param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            print("Training logistic regression model...")
            self.model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='liblinear',
                class_weight='balanced'
            )
            self.model.fit(self.X_train_scaled, self.y_train)
        
        return self
    
    def evaluate_model(self):
        """
        Evaluate the model performance
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        if hasattr(self.model, 'predict') and hasattr(self.model, 'predict_proba'):
            y_pred = self.model.predict(self.X_test_scaled)
            y_pred_proba = self.model.predict_proba(self.X_test_scaled)
        else:
            raise ValueError("Model does not have required predict methods")
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Handle AUC calculation for small datasets
        try:
            auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
        except ValueError:
            auc_score = None
            print("Warning: Could not calculate AUC score (insufficient data)")
        
        print("=" * 50)
        print("MODEL PERFORMANCE")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        if auc_score is not None:
            print(f"AUC Score: {auc_score:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                 target_names=['Left-leaning', 'Right-leaning']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self._plot_confusion_matrix(cm)
        
        # ROC curve (only if we have enough data)
        if auc_score is not None:
            self._plot_roc_curve(y_pred_proba[:, 1])
        
        # Precision-Recall curve
        self._plot_precision_recall_curve(y_pred_proba[:, 1])
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def analyze_feature_importance(self, top_n=15):
        """
        Analyze and plot feature importance
        """
        if self.model is None:
            raise ValueError("Model must be trained before analyzing features")
        
        # Get feature importance (absolute coefficients)
        feature_importance = np.abs(self.model.coef_[0])
        
        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance,
            'coefficient': self.model.coef_[0]
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print("\n" + "=" * 50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        print(f"Top {top_n} most important features:")
        print(importance_df.head(top_n)[['feature', 'importance', 'coefficient']])
        
        # Plot feature importance
        self._plot_feature_importance(importance_df.head(top_n))
        
        # Analyze feature patterns
        self._analyze_feature_patterns(importance_df)
        
        return importance_df
    
    def cross_validate(self, cv=None):
        """
        Perform cross-validation
        """
        # Adjust CV folds for small datasets
        if cv is None:
            cv = min(3, len(self.X_train) // 2)
        
        try:
            scores = cross_val_score(
                self.model, self.X_train_scaled, self.y_train, 
                cv=cv, scoring='accuracy'
            )
            
            print(f"\nCross-validation scores: {scores}")
            print(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            return scores
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            print("Dataset too small for cross-validation")
            return None
    
    def predict_new_data(self, new_data):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale the new data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make predictions
        predictions = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)
        
        return predictions, probabilities
    
    def _plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Left-leaning', 'Right-leaning'],
                   yticklabels=['Left-leaning', 'Right-leaning'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def _plot_roc_curve(self, y_pred_proba):
        """
        Plot ROC curve
        """
        try:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.show()
        except Exception as e:
            print(f"Could not plot ROC curve: {e}")
    
    def _plot_precision_recall_curve(self, y_pred_proba):
        """
        Plot Precision-Recall curve
        """
        try:
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Could not plot Precision-Recall curve: {e}")
    
    def _plot_feature_importance(self, importance_df):
        """
        Plot feature importance
        """
        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'blue' for x in importance_df['coefficient']]
        
        plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance (Absolute Coefficient)')
        plt.title('Top Features by Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def _analyze_feature_patterns(self, importance_df):
        """
        Analyze patterns in feature importance
        """
        print("\nFeature Pattern Analysis:")
        
        # Top positive coefficients (right-leaning indicators)
        positive_features = importance_df[importance_df['coefficient'] > 0].head(5)
        print("\nTop features indicating RIGHT-leaning (positive coefficients):")
        for _, row in positive_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        # Top negative coefficients (left-leaning indicators)
        negative_features = importance_df[importance_df['coefficient'] < 0].head(5)
        print("\nTop features indicating LEFT-leaning (negative coefficients):")
        for _, row in negative_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        # Feature categories analysis
        immigration_features = [f for f in importance_df['feature'] if any(word in f.lower() for word in ['immigration', 'border', 'deport', 'asylum'])]
        legal_features = [f for f in importance_df['feature'] if any(word in f.lower() for word in ['law', 'rights', 'constitutional', 'judge', 'lawsuit'])]
        admin_features = [f for f in importance_df['feature'] if any(word in f.lower() for word in ['administration', 'trump', 'biden'])]
        
        print(f"\nFeature Categories:")
        print(f"  Immigration-related features: {len(immigration_features)}")
        print(f"  Legal/constitutional features: {len(legal_features)}")
        print(f"  Administration-related features: {len(admin_features)}")

def main():
    """
    Main function to run the complete analysis
    """
    print("=" * 60)
    print("POLITICAL LEANING CLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PoliticalLeaningAnalyzer('Scores.csv')
    
    # Load and prepare data
    analyzer.load_data()
    analyzer.split_data()
    
    # Train model (skip hyperparameter tuning for small dataset)
    analyzer.train_model(tune_hyperparameters=False)
    
    # Evaluate model
    results = analyzer.evaluate_model()
    
    # Analyze feature importance
    importance_df = analyzer.analyze_feature_importance(top_n=15)
    
    # Cross-validation (if possible)
    cv_scores = analyzer.cross_validate()
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Model Accuracy: {results['accuracy']:.4f}")
    if results['auc_score'] is not None:
        print(f"Model AUC: {results['auc_score']:.4f}")
    if cv_scores is not None:
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\nTop 5 most important features:")
    top_features = importance_df.head(5)
    for _, row in top_features.iterrows():
        leaning = "RIGHT-leaning" if row['coefficient'] > 0 else "LEFT-leaning"
        print(f"  {row['feature']}: {row['importance']:.4f} ({leaning})")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 