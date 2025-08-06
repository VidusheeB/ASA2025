import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import random

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

def train_model_on_sample(X_sample, y_sample):
    """Train a logistic regression model on a sample"""
    # Split the sample into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
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
    
    return y_test, y_pred

def analyze_predictions(y_true, y_pred):
    """Analyze prediction patterns"""
    results = {
        'correct': 0,
        'left_to_right': 0,  # Actual left, predicted right
        'right_to_left': 0,  # Actual right, predicted left
        'total': len(y_true)
    }
    
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            results['correct'] += 1
        elif true == 0 and pred == 1:  # Left predicted as right
            results['left_to_right'] += 1
        elif true == 1 and pred == 0:  # Right predicted as left
            results['right_to_left'] += 1
    
    # Calculate proportions
    results['correct_proportion'] = results['correct'] / results['total']
    results['left_to_right_proportion'] = results['left_to_right'] / results['total']
    results['right_to_left_proportion'] = results['right_to_left'] / results['total']
    
    return results

def run_sample_analysis():
    """Run the complete sample analysis"""
    print("=" * 60)
    print("RANDOM SAMPLE ANALYSIS")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Store results for all samples
    all_results = []
    
    # Run 40 random samples
    for sample_num in range(1, 41):
        print(f"\nSample {sample_num}/40")
        
        # Take random sample of 10 articles
        sample_indices = random.sample(range(len(X)), 10)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        # Train model and get predictions
        y_test, y_pred = train_model_on_sample(X_sample, y_sample)
        
        # Analyze predictions
        results = analyze_predictions(y_test, y_pred)
        results['sample_num'] = sample_num
        
        all_results.append(results)
        
        print(f"  Correct: {results['correct']}/{results['total']} ({results['correct_proportion']:.2%})")
        print(f"  Left→Right: {results['left_to_right']} ({results['left_to_right_proportion']:.2%})")
        print(f"  Right→Left: {results['right_to_left']} ({results['right_to_left_proportion']:.2%})")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (40 samples)")
    print("=" * 60)
    
    correct_proportions = [r['correct_proportion'] for r in all_results]
    left_to_right_proportions = [r['left_to_right_proportion'] for r in all_results]
    right_to_left_proportions = [r['right_to_left_proportion'] for r in all_results]
    
    print(f"Average Correct Proportion: {np.mean(correct_proportions):.3f} (±{np.std(correct_proportions):.3f})")
    print(f"Average Left→Right Proportion: {np.mean(left_to_right_proportions):.3f} (±{np.std(left_to_right_proportions):.3f})")
    print(f"Average Right→Left Proportion: {np.mean(right_to_left_proportions):.3f} (±{np.std(right_to_left_proportions):.3f})")
    
    # Find best and worst performing samples
    best_sample = max(all_results, key=lambda x: x['correct_proportion'])
    worst_sample = min(all_results, key=lambda x: x['correct_proportion'])
    
    print(f"\nBest Sample (#{best_sample['sample_num']}): {best_sample['correct_proportion']:.2%} correct")
    print(f"Worst Sample (#{worst_sample['sample_num']}): {worst_sample['correct_proportion']:.2%} correct")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('sample_analysis_results.csv', index=False)
    print(f"\nDetailed results saved to: sample_analysis_results.csv")
    
    return all_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    run_sample_analysis() 