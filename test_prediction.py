#!/usr/bin/env python3
"""
Test script for political leaning prediction
"""

from predict import predict_from_text

def test_predictions():
    """Test predictions with sample articles"""
    
    test_articles = [
        {
            "title": "Conservative Immigration Article",
            "text": "The Trump administration has announced new deportation plans that will target undocumented immigrants across the country. The executive order calls for increased border security and ICE operations to enforce immigration law."
        },
        {
            "title": "Liberal Immigration Article", 
            "text": "Civil rights groups including the ACLU have filed lawsuits challenging the constitutionality of immigration measures, citing concerns about racial profiling and violations of the Fourth Amendment."
        }
    ]
    
    print("Testing Political Leaning Predictions")
    print("=" * 50)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n📰 Article {i}: {article['title']}")
        print("-" * 30)
        
        prediction, probability, features, feature_analysis = predict_from_text(article['text'])
        
        if prediction is not None and probability is not None:
            if prediction == 0:
                print("🎯 Prediction: LEFT-LEANING 🟦")
            else:
                print("🎯 Prediction: RIGHT-LEANING 🟥")
            
            print(f"Confidence: {max(probability):.3f}")
            print(f"Probabilities: Left={probability[0]:.3f}, Right={probability[1]:.3f}")
            
            if feature_analysis:
                print("\nTop features found:")
                for j, feature in enumerate(feature_analysis[:3]):
                    leaning = "RIGHT" if feature['coefficient'] > 0 else "LEFT"
                    print(f"  {j+1}. {feature['feature']}: {feature['score']:.4f} ({leaning}-leaning)")
        else:
            print("❌ Could not make prediction")
        
        print()

if __name__ == "__main__":
    test_predictions() 