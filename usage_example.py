#!/usr/bin/env python3
"""
Usage Example for Political Leaning Classifier

This script demonstrates how to use the trained model to predict
political leaning of articles.
"""

from predict import predict_from_text

def main():
    """Example usage of the political leaning predictor"""
    
    # Example articles for testing
    example_articles = [
        {
            "title": "Conservative Immigration Article",
            "text": "The Trump administration has announced new deportation plans that will target undocumented immigrants across the country. The executive order calls for increased border security and ICE operations to enforce immigration law. The Biden administration's sanctuary policies undermine border security and encourage illegal immigration."
        },
        {
            "title": "Liberal Immigration Article", 
            "text": "Civil rights groups including the ACLU have filed lawsuits challenging the constitutionality of immigration measures, citing concerns about racial profiling and violations of the Fourth Amendment. The Biden administration has implemented sanctuary policies in some states to protect human rights and constitutional protections."
        },
        {
            "title": "Neutral Article",
            "text": "The debate over immigration reform continues to divide the country, with various perspectives on border security, human rights, and economic impacts. Different administrations have implemented various policies to address these complex issues."
        }
    ]
    
    print("=" * 60)
    print("POLITICAL LEANING CLASSIFIER - USAGE EXAMPLE")
    print("=" * 60)
    
    for i, article in enumerate(example_articles, 1):
        print(f"\n📰 Article {i}: {article['title']}")
        print("-" * 40)
        
        # Make prediction
        prediction, probability, features, feature_analysis = predict_from_text(article['text'])
        
        if prediction is not None and probability is not None:
            # Show prediction
            if prediction == 0:
                print("🎯 Prediction: LEFT-LEANING 🟦")
            else:
                print("🎯 Prediction: RIGHT-LEANING 🟥")
            
            print(f"Confidence: {max(probability):.3f}")
            
            # Show probabilities
            print(f"Probabilities:")
            print(f"  Left-leaning: {probability[0]:.3f}")
            print(f"  Right-leaning: {probability[1]:.3f}")
            
            # Show top features
            if feature_analysis:
                print(f"\nTop features found:")
                for j, feature in enumerate(feature_analysis[:5]):
                    leaning = "RIGHT" if feature['coefficient'] > 0 else "LEFT"
                    print(f"  {j+1}. {feature['feature']}: {feature['score']:.4f} ({leaning}-leaning)")
        else:
            print("❌ Could not make prediction (model not loaded)")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 