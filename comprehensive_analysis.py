import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent pop-ups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve
)
import os
import pickle
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnalysis:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.results = {}
        self.datasets = {}
        
    def load_all_datasets(self):
        """Load all available TF-IDF datasets from the data directory."""
        datasets = {}
        
        # Look for all CSV files in the data directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv') and 'TF-IDF' in filename:
                file_path = os.path.join(self.data_dir, filename)
                
                # Extract set number and word count from filename
                if 'Set 1' in filename:
                    set_num = 1
                elif 'Set 2' in filename:
                    set_num = 2
                elif 'Set 3' in filename:
                    set_num = 3
                else:
                    continue
                
                if 'Ready 20' in filename:
                    word_count = 20
                elif 'Ready 25' in filename:
                    word_count = 25
                elif 'Ready 30' in filename:
                    word_count = 30
                else:
                    continue
                
                # Load the dataset
                try:
                    data = pd.read_csv(file_path)
                    print(f"Loaded: {filename}")
                    datasets[f"Set{set_num}_{word_count}w"] = {
                        'data': data,
                        'filename': filename,
                        'set_num': set_num,
                        'word_count': word_count
                    }
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        # Also check for the new Set 3 files without "Ready" in the name
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv') and 'Set 3' in filename and 'TF-IDF' not in filename:
                file_path = os.path.join(self.data_dir, filename)
                
                # Extract word count from filename
                if '20 Words' in filename:
                    word_count = 20
                elif '25 Words' in filename:
                    word_count = 25
                elif '30 Words' in filename:
                    word_count = 30
                else:
                    continue
                
                # Load the dataset
                try:
                    data = pd.read_csv(file_path)
                    print(f"Loaded: {filename}")
                    datasets[f"Set3_{word_count}w"] = {
                        'data': data,
                        'filename': filename,
                        'set_num': 3,
                        'word_count': word_count
                    }
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if not datasets:
            raise ValueError("No valid TF-IDF datasets found in the data directory")
        
        print(f"\nLoaded {len(datasets)} datasets:")
        for key, info in datasets.items():
            print(f"  {key}: {info['filename']} ({info['word_count']} words)")
        
        return datasets
    
    def perform_proper_kfold_analysis(self, X, y, k, dataset_name):
        """Perform proper k-fold cross-validation with TF-IDF calculated per fold"""
        print(f"\n--- Analyzing {dataset_name} with Proper K={k} Cross-Validation ---")
        
        # Initialize k-fold
        if k == 1:
            # Leave-one-out - use regular KFold for LOO
            kfold = KFold(n_splits=len(y), shuffle=True, random_state=42)
            use_stratified = False
        else:
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            use_stratified = True
        
        # Storage for predictions
        all_predictions = np.zeros(len(y), dtype=int)
        all_probabilities = np.zeros(len(y), dtype=float)
        fold_scores = []
        fold_thresholds = []
        
        # Get feature names for TF-IDF
        feature_names = X.columns.tolist()
        
        if use_stratified:
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
                print(f"  Processing fold {fold_idx}/{k}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Initialize model
                model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Get predictions
                train_proba = model.predict_proba(X_train_scaled)[:, 1]
                test_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Find optimal threshold on training set
                optimal_thresholds = self.find_optimal_threshold(y_train, train_proba)
                optimal_threshold = optimal_thresholds['f1_threshold']
                fold_thresholds.append(optimal_threshold)
                
                # Make predictions with optimal threshold
                test_predictions = (test_proba >= optimal_threshold).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, test_predictions)
                fold_scores.append(accuracy)
                
                # Store predictions
                all_predictions[test_idx] = test_predictions
                all_probabilities[test_idx] = test_proba
                
                print(f"    Fold {fold_idx} Accuracy: {accuracy:.3f} (threshold: {optimal_threshold:.3f})")
        else:
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
                print(f"  Processing fold {fold_idx}/{len(y)}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Initialize model
                model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Get predictions
                train_proba = model.predict_proba(X_train_scaled)[:, 1]
                test_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Find optimal threshold on training set
                optimal_thresholds = self.find_optimal_threshold(y_train, train_proba)
                optimal_threshold = optimal_thresholds['f1_threshold']
                fold_thresholds.append(optimal_threshold)
                
                # Make predictions with optimal threshold
                test_predictions = (test_proba >= optimal_threshold).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, test_predictions)
                fold_scores.append(accuracy)
                
                # Store predictions
                all_predictions[test_idx] = test_predictions
                all_probabilities[test_idx] = test_proba
                
                print(f"    Fold {fold_idx} Accuracy: {accuracy:.3f} (threshold: {optimal_threshold:.3f})")
        
        # Calculate overall metrics
        overall_accuracy = accuracy_score(y, all_predictions)
        cm = confusion_matrix(y, all_predictions)
        
        # Calculate sensitivity and specificity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        result = {
            'k': k,
            'dataset_name': dataset_name,
            'accuracy': overall_accuracy,
            'cv_scores': np.array(fold_scores),
            'cv_accuracy_mean': np.mean(fold_scores),
            'cv_accuracy_std': np.std(fold_scores),
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'confusion_matrix': cm,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'optimal_thresholds': {'f1_threshold': np.mean(fold_thresholds)},
            'optimal_threshold_used': np.mean(fold_thresholds)
        }
        
        print(f"Overall Accuracy: {overall_accuracy:.3f} (±{np.std(fold_scores):.3f})")
        print(f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
        
        return result
    
    def find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold for given predictions"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal thresholds
        optimal_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        optimal_f1 = results_df.loc[results_df['f1_score'].idxmax()]
        optimal_precision = results_df.loc[results_df['precision'].idxmax()]
        optimal_recall = results_df.loc[results_df['recall'].idxmax()]
        
        return {
            'accuracy_threshold': optimal_accuracy['threshold'],
            'f1_threshold': optimal_f1['threshold'],
            'precision_threshold': optimal_precision['threshold'],
            'recall_threshold': optimal_recall['threshold'],
            'results_df': results_df
        }
    

    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis"""
        print("=" * 80)
        print("COMPREHENSIVE TF-IDF ANALYSIS")
        print("=" * 80)
        
        # Load all datasets
        self.datasets = self.load_all_datasets()
        
        # Define configurations
        k_values = [5, 10, 1]  # 1 fold means leave-one-out
        
        # Run analysis for all combinations
        for dataset_name, dataset_data in self.datasets.items():
            data = dataset_data['data'] # Use 'data' from the new structure
            print(f"Processing {dataset_name}, columns: {list(data.columns)}")
            
            # Handle different column names for target
            if 'Leaning' in data.columns:
                y = data['Leaning']  # Target column
                X = data.drop('Leaning', axis=1)  # Feature columns
            elif 'Document Name' in data.columns:
                # For Set 2 files, the first column contains the target values
                y = data.iloc[:, 0]  # First column as target
                X = data.drop(data.columns[0], axis=1)  # Drop first column
            else:
                print(f"Warning: No target column found in {dataset_name}")
                continue
            
            # Clean data - remove rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                print(f"Warning: No valid data after cleaning for {dataset_name}")
                continue
                
            feature_names = X.columns.tolist()
            
            for k in k_values:
                if k == 1:
                    # Leave-one-out cross-validation
                    config_name = f"{dataset_name}_LOO"
                else:
                    config_name = f"{dataset_name}_K{k}"
                
                # Use the proper k-fold analysis method
                result = self.perform_proper_kfold_analysis(X, y, k, dataset_name)
                self.results[config_name] = result
        
        # Generate visualizations and save results
        self.create_visualizations()
        self.save_results()
        
        return self.results

    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create directory for results
        os.makedirs('comprehensive_results', exist_ok=True)
        
        # 1. Confusion Matrices
        self.create_confusion_matrices()
        
        # 2. Accuracy Comparison
        self.create_accuracy_comparison()
        
        # 3. Average Accuracy Comparison
        self.create_average_accuracy_comparison()
        
        # 4. Performance Summary
        self.create_performance_summary()
    
    def create_confusion_matrices(self):
        """Create confusion matrices for all configurations"""
        print("Creating confusion matrices...")
        
        n_configs = len(self.results)
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (config_name, result) in enumerate(self.results.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Left', 'Right'], 
                       yticklabels=['Left', 'Right'],
                       ax=axes[row, col])
            
            # Calculate sensitivity and specificity
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            axes[row, col].set_title(f'{config_name}\nAcc: {result["accuracy"]:.3f}, Sens: {sensitivity:.3f}, Spec: {specificity:.3f}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_configs, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('comprehensive_results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrices saved to: comprehensive_results/confusion_matrices.png")
    
    def create_accuracy_comparison(self):
        """Create accuracy comparison plot"""
        print("Creating accuracy comparison...")
        
        config_names = list(self.results.keys())
        accuracies = [self.results[config]['accuracy'] for config in config_names]
        cv_means = [self.results[config]['cv_accuracy_mean'] for config in config_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(config_names, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Accuracy Comparison Across Configurations')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # CV Accuracy Mean comparison
        bars2 = ax2.bar(config_names, cv_means, color='lightgreen', alpha=0.7)
        ax2.set_title('Cross-Validation Accuracy Mean Comparison')
        ax2.set_ylabel('CV Accuracy Mean')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, cv_mean in zip(bars2, cv_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{cv_mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('comprehensive_results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Accuracy comparison saved to: comprehensive_results/accuracy_comparison.png")
    
    def create_average_accuracy_comparison(self):
        """Create average accuracy comparison plot by word count and set"""
        print("Creating average accuracy comparison...")
        
        # Organize results by set and word count
        set_data = {}
        
        for config_name, result in self.results.items():
            # Parse configuration name to extract set and word count
            if 'Set1' in config_name:
                set_name = 'Set 1'
                if '20w' in config_name:
                    word_count = 20
                elif '25w' in config_name:
                    word_count = 25
                elif '30w' in config_name:
                    word_count = 30
            elif 'Set2' in config_name:
                set_name = 'Set 2'
                if '20w' in config_name:
                    word_count = 20
                elif '25w' in config_name:
                    word_count = 25
                elif '30w' in config_name:
                    word_count = 30
            elif 'Set3' in config_name:
                set_name = 'Set 3'
                if '20w' in config_name:
                    word_count = 20
                elif '25w' in config_name:
                    word_count = 25
                elif '30w' in config_name:
                    word_count = 30
            else:
                continue
            
            if set_name not in set_data:
                set_data[set_name] = {}
            if word_count not in set_data[set_name]:
                set_data[set_name][word_count] = []
            
            set_data[set_name][word_count].append(result['accuracy'])
        
        # Calculate averages
        averages = {}
        for set_name, word_counts in set_data.items():
            averages[set_name] = {}
            for word_count, accuracies in word_counts.items():
                averages[set_name][word_count] = np.mean(accuracies)
        
        # Create the line plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        word_counts = [20, 25, 30]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        set_names = ['Set 1', 'Set 2', 'Set 3']
        
        for i, set_name in enumerate(set_names):
            if set_name in averages:
                accuracies = [averages[set_name].get(wc, 0) for wc in word_counts]
                line = ax.plot(word_counts, accuracies, 'o-', 
                             label=set_name, color=colors[i], linewidth=3, markersize=8)
                
                # Add value labels on points
                for x, y in zip(word_counts, accuracies):
                    ax.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Number of Words', fontsize=12)
        ax.set_ylabel('Average Accuracy', fontsize=12)
        ax.set_title('Average Accuracy by Word Count and Set\n(5-fold, 10-fold, and LOO combined)', fontsize=14, fontweight='bold')
        ax.set_xticks(word_counts)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add summary statistics
        summary_text = "Summary:\n"
        for set_name in set_names:
            if set_name in averages:
                best_wc = max(averages[set_name].items(), key=lambda x: x[1])
                summary_text += f"{set_name}: Best at {best_wc[0]} words ({best_wc[1]:.3f})\n"
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('comprehensive_results/average_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Average accuracy comparison saved to: comprehensive_results/average_accuracy_comparison.png")
        
        # Print summary to console
        print("\n" + "=" * 50)
        print("AVERAGE ACCURACY SUMMARY")
        print("=" * 50)
        for set_name in set_names:
            if set_name in averages:
                print(f"\n{set_name}:")
                for word_count in word_counts:
                    if word_count in averages[set_name]:
                        print(f"  {word_count} words: {averages[set_name][word_count]:.3f}")
                best_wc = max(averages[set_name].items(), key=lambda x: x[1])
                print(f"  Best: {best_wc[0]} words ({best_wc[1]:.3f})")
        
        # Find overall best
        all_averages = []
        for set_name in set_names:
            if set_name in averages:
                for word_count, acc in averages[set_name].items():
                    all_averages.append((set_name, word_count, acc))
        
        if all_averages:
            best_overall = max(all_averages, key=lambda x: x[2])
            print(f"\nOverall Best: {best_overall[0]} with {best_overall[1]} words ({best_overall[2]:.3f})")
        print("=" * 50)
    
    def create_threshold_analysis(self):
        """Create optimal threshold analysis"""
        print("Creating threshold analysis...")
        
        n_configs = len(self.results)
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (config_name, result) in enumerate(self.results.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            results_df = result['optimal_thresholds']['results_df']
            
            ax = axes[row, col]
            ax.plot(results_df['threshold'], results_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
            ax.plot(results_df['threshold'], results_df['f1_score'], 'r-', label='F1-Score', linewidth=2)
            ax.plot(results_df['threshold'], results_df['precision'], 'g-', label='Precision', linewidth=2)
            ax.plot(results_df['threshold'], results_df['recall'], 'm-', label='Recall', linewidth=2)
            
            # Mark optimal threshold
            optimal_threshold = result['optimal_threshold_used']
            ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'Optimal F1: {optimal_threshold:.3f}')
            
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title(f'{config_name}\nThreshold Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_configs, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('comprehensive_results/threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Threshold analysis saved to: comprehensive_results/threshold_analysis.png")
    

    
    def create_performance_summary(self):
        """Create performance summary table"""
        print("Creating performance summary...")
        
        summary_data = []
        for config_name, result in self.results.items():
            summary_data.append({
                'Configuration': config_name,
                'Accuracy': f"{result['accuracy']:.3f}",
                'CV_Accuracy_Mean': f"{result['cv_accuracy_mean']:.3f}",
                'CV_Accuracy_Std': f"{result['cv_accuracy_std']:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('comprehensive_results/performance_summary.csv', index=False)
        print("Performance summary saved to: comprehensive_results/performance_summary.csv")
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        configs = summary_df['Configuration']
        accuracies = [float(x) for x in summary_df['Accuracy']]
        
        x = np.arange(len(configs))
        
        bars = ax.bar(x, accuracies, color='skyblue', alpha=0.7)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Summary Across All Configurations')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('comprehensive_results/performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Performance summary plot saved to: comprehensive_results/performance_summary.png")
    
    def save_results(self):
        """Save detailed results"""
        print("\nSaving detailed results...")
        
        # Save all results to pickle file
        with open('comprehensive_results/analysis_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save datasets info
        datasets_info = {}
        for name, data in self.datasets.items():
            datasets_info[name] = {
                'feature_names': data['data'].columns.tolist(), # Use 'data' from the new structure
                'word_count': data['word_count'],
                'file_path': data['filename']
            }
        
        with open('comprehensive_results/datasets_info.pkl', 'wb') as f:
            pickle.dump(datasets_info, f)
        
        print("Detailed results saved to comprehensive_results/")
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Find best configurations
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"\nBest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.3f})")
        
        print(f"\nDetailed Results:")
        for config_name, result in self.results.items():
            print(f"\n{config_name}:")
            print(f"  Accuracy: {result['accuracy']:.3f} (±{result['cv_accuracy_std']:.3f})")

def main():
    """Main function to run comprehensive analysis"""
    analyzer = ComprehensiveAnalysis()
    results = analyzer.run_comprehensive_analysis()
    analyzer.print_summary()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("All results and visualizations saved to 'comprehensive_results/' directory")
    print("You can now use the Streamlit app to view the results interactively")

if __name__ == "__main__":
    main()
