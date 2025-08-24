#!/usr/bin/env python3
"""
Standalone script to run K-fold evaluation on the fine-tuned political classifier.
"""

import os
import sys
from dotenv import load_dotenv
from fine_tuned_classifier import FineTunedPoliticalClassifier
from kfold_evaluator import KFoldEvaluator

load_dotenv()

def main():
    """Run K-fold evaluation with command line interface."""
    print("🏛️ Political Leaning Classifier - K-Fold Evaluation")
    print("=" * 60)
    
    # Check for required environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    model_name = os.getenv('FINE_TUNED_MODEL_NAME')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    if not model_name:
        print("❌ Error: FINE_TUNED_MODEL_NAME environment variable not set")
        print("Please set your fine-tuned model name:")
        print("export FINE_TUNED_MODEL_NAME='ft:gpt-4.1:your-model-id'")
        sys.exit(1)
    
    # Initialize classifier
    print("\n🔧 Initializing classifier...")
    try:
        classifier = FineTunedPoliticalClassifier(model_name=model_name, api_key=api_key)
        print("✅ Classifier initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        sys.exit(1)
    
    # Check for training data
    training_file = "political_leaning_training.jsonl"
    if not os.path.exists(training_file):
        print(f"❌ Error: Training data file '{training_file}' not found")
        print("Please ensure the JSONL training file is in the current directory")
        sys.exit(1)
    
    # Get K-fold parameter
    try:
        k_folds = int(input("\nEnter number of folds (default: 5): ") or "5")
        if k_folds < 2 or k_folds > 20:
            print("❌ Error: Number of folds must be between 2 and 20")
            sys.exit(1)
    except ValueError:
        print("❌ Error: Invalid number of folds")
        sys.exit(1)
    
    # Initialize evaluator
    print(f"\n🔬 Initializing {k_folds}-fold evaluator...")
    evaluator = KFoldEvaluator(classifier, k_folds=k_folds)
    
    # Load training data
    print("📊 Loading training data...")
    try:
        tfidf_data, labels = evaluator.load_training_data(training_file)
        print(f"✅ Loaded {len(tfidf_data)} samples with {len(set(labels))} classes")
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        sys.exit(1)
    
    # Perform evaluation
    print(f"\n🚀 Starting {k_folds}-fold cross-validation...")
    print("This may take several minutes depending on your data size and API rate limits.")
    
    try:
        results = evaluator.perform_kfold_evaluation(tfidf_data, labels)
        print("✅ K-fold evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)
    
    # Print summary
    evaluator.print_summary()
    
    # Generate plots and save results
    print("\n📈 Generating visualizations and saving results...")
    try:
        evaluator.save_results_to_csv("kfold_results.csv")
        evaluator.generate_confusion_matrix_plot("kfold_confusion_matrix.png")
        evaluator.generate_fold_performance_plot("kfold_performance.png")
        print("✅ Results saved to files:")
        print("   - kfold_results.csv")
        print("   - kfold_confusion_matrix.png")
        print("   - kfold_performance.png")
    except Exception as e:
        print(f"⚠️ Warning: Error saving results: {e}")
    
    print("\n🎉 Evaluation complete!")
    print("You can now run the web interface with: streamlit run app.py")

if __name__ == "__main__":
    main() 