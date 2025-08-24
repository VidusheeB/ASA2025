#!/usr/bin/env python3
"""
Simple setup script for the Political Leaning Classifier.
"""

import os

def create_env_file():
    """Create a .env file with user input."""
    print("🏛️ Political Leaning Classifier - Setup")
    print("=" * 50)
    
    print("\nPlease provide your configuration:")
    
    # Get API key
    api_key = input("\n1. OpenAI API Key (required): ").strip()
    if not api_key:
        print("❌ API key is required!")
        return False
    
    # Get fine-tuned model name
    print("\n2. Fine-tuned Model Name (required)")
    print("   Format: ft:gpt-4.1:your-model-id")
    print("   This is required - no fallback available")
    model_name = input("   Enter your fine-tuned model name: ").strip()
    
    # Validate model name
    if not model_name:
        print("   ❌ Error: Fine-tuned model name is required")
        return False
    
    # Create .env file
    env_content = f"# Political Leaning Classifier Configuration\nOPENAI_API_KEY={api_key}\n"
    env_content += f"FINE_TUNED_MODEL_NAME={model_name}\n"
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Configuration saved to .env file")
    print(f"   API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if len(api_key) > 8 else '*' * len(api_key)}")
    print(f"   Model: {model_name}")
    
    return True

if __name__ == "__main__":
    if create_env_file():
        print("\n🎉 Setup completed!")
        print("\nNext steps:")
        print("1. Test the system: python test_system.py")
        print("2. Run web interface: streamlit run app.py")
        print("3. Run K-fold evaluation: python run_kfold_evaluation.py")
    else:
        print("\n❌ Setup failed. Please try again.") 