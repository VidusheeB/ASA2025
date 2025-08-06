import pandas as pd
import json

def create_tfidf_jsonl():
    """Create JSONL file with TF-IDF scores and terms for OpenAI fine-tuning"""
    
    print("Creating JSONL file with TF-IDF scores and terms...")
    
    # Load the CSV data
    data = pd.read_csv('Scores.csv')
    
    # Get feature names (all columns except 'Label')
    feature_names = [col for col in data.columns if col != 'Label']
    
    print(f"Found {len(data)} samples with {len(feature_names)} features")
    
    # Create JSONL file
    jsonl_data = []
    
    for idx, row in data.iterrows():
        label = row['Label']
        features = row.drop('Label')
        
        # Create feature dictionary with all scores (including zeros)
        tfidf_scores = {}
        for feature_name in feature_names:
            score = features[feature_name]
            tfidf_scores[feature_name] = score
        
        # Determine political leaning
        leaning = "left-leaning" if label == 0 else "right-leaning"
        
        # Create training example
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a political leaning classifier. Analyze the TF-IDF scores for political terms and predict whether an article is left-leaning or right-leaning."
                },
                {
                    "role": "user", 
                    "content": f"TF-IDF scores: {json.dumps(tfidf_scores, indent=2)}"
                },
                {
                    "role": "assistant",
                    "content": f"This article is {leaning}."
                }
            ]
        }
        
        jsonl_data.append(training_example)
    
    # Write to JSONL file
    output_file = 'political_leaning_training.jsonl'
    with open(output_file, 'w') as f:
        for example in jsonl_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created {output_file} with {len(jsonl_data)} training examples")
    print(f"Political leaning distribution:")
    print(f"- Left-leaning: {sum(1 for ex in jsonl_data if 'left-leaning' in ex['messages'][-1]['content'])}")
    print(f"- Right-leaning: {sum(1 for ex in jsonl_data if 'right-leaning' in ex['messages'][-1]['content'])}")
    
    # Show sample of the data
    print(f"\nSample training example:")
    print(json.dumps(jsonl_data[0], indent=2))
    
    return output_file

if __name__ == "__main__":
    create_tfidf_jsonl() 