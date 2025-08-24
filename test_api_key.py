#!/usr/bin/env python3
"""
Test script to verify API key and model access
"""

import os
import streamlit as st
from openai import OpenAI

def test_api_key():
    """Test if the API key works and can access the model"""
    
    # Get API key from Streamlit secrets
    try:
        api_key = st.secrets.get('OPENAI_API_KEY')
        model_name = st.secrets.get('FINE_TUNED_MODEL_NAME')
        
        print(f"🔑 API Key found: {api_key[:20]}...")
        print(f"🤖 Model Name: {model_name}")
        
        # Test basic API connection
        client = OpenAI(api_key=api_key)
        
        # Test 1: List models to see if API key works
        print("\n📋 Testing API key with model list...")
        try:
            models = client.models.list()
            print(f"✅ API key works! Found {len(models.data)} models")
            
            # Check if our fine-tuned model is in the list
            model_ids = [model.id for model in models.data]
            if model_name in model_ids:
                print(f"✅ Fine-tuned model '{model_name}' found in accessible models!")
            else:
                print(f"❌ Fine-tuned model '{model_name}' NOT found in accessible models")
                print("Available models (first 10):")
                for i, model_id in enumerate(model_ids[:10]):
                    print(f"  {i+1}. {model_id}")
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return False
            
        # Test 2: Try a simple chat completion
        print("\n💬 Testing simple chat completion...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10
            )
            print(f"✅ Simple chat completion works: {response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ Error with simple chat completion: {e}")
            return False
            
        # Test 3: Try the fine-tuned model
        print(f"\n🎯 Testing fine-tuned model '{model_name}'...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            print(f"✅ Fine-tuned model works: {response.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"❌ Error with fine-tuned model: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing secrets: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing API Key and Model Access")
    print("=" * 50)
    
    success = test_api_key()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Your setup should work.")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\n💡 Troubleshooting tips:")
        print("1. Verify your API key at https://platform.openai.com/account/api-keys")
        print("2. Make sure your API key has access to fine-tuned models")
        print("3. Check if your fine-tuned model is still active")
        print("4. Try creating a new API key")
