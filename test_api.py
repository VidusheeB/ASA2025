#!/usr/bin/env python3
"""
Test script to verify OpenAI API key and model access.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_api_key():
    """Test the OpenAI API key."""
    print("🔑 Testing OpenAI API Key...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found in .env file")
        return False
    
    print(f"✅ API key found: {api_key[:20]}...")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test with the fine-tuned model
        model_name = os.getenv('FINE_TUNED_MODEL_NAME')
        if not model_name:
            print("❌ Error: FINE_TUNED_MODEL_NAME environment variable not set")
            return False
            
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✅ API key is valid and working!")
        return True
        
    except Exception as e:
        print(f"❌ API key error: {e}")
        return False

def test_fine_tuned_model():
    """Test access to the fine-tuned model."""
    print("\n🤖 Testing Fine-tuned Model...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    model_name = os.getenv('FINE_TUNED_MODEL_NAME')
    
    if not model_name:
        print("❌ Error: FINE_TUNED_MODEL_NAME environment variable not set")
        print("Please set your fine-tuned model name:")
        print("export FINE_TUNED_MODEL_NAME='ft:gpt-4.1:your-model-id'")
        return False
    
    print(f"✅ Model name: {model_name}")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test with the fine-tuned model
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a political leaning classifier."},
                {"role": "user", "content": "TF-IDF scores: {\"Scrutiny\": 0.0, \"national security\": 1.5}"}
            ],
            max_tokens=50
        )
        
        print("✅ Fine-tuned model is accessible!")
        print(f"   Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Fine-tuned model error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 OpenAI API Test")
    print("=" * 40)
    
    if test_api_key():
        test_fine_tuned_model()
    else:
        print("\n❌ Please check your API key and try again.") 