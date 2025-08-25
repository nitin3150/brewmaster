# gemini_test.py - Simple script to test your Gemini API key

import os
import json
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()
# Try to import Gemini
try:
    import google.generativeai as genai
    print("✅ google-generativeai library is installed")
except ImportError:
    print("❌ google-generativeai not installed")
    print("💡 Install with: pip install google-generativeai")
    exit(1)

def test_gemini_api_key():
    """Test Gemini API key with different models"""
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("💡 Set it with: export GEMINI_API_KEY='your_key_here'")
        print("💡 Or create a .env file with: GEMINI_API_KEY=your_key_here")
        return False
    
    if api_key == "your_gemini_api_key_here":
        print("❌ Please set a real API key (not the placeholder)")
        return False
    
    print(f"🔑 API Key found: {api_key[:10]}...")
    
    # Configure Gemini
    try:
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured")
    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {e}")
        return False
    
    # Test different models
    models_to_test = [
        'gemini-1.5-flash',
        'gemini-1.5-pro', 
        'gemini-pro-latest',
        'gemini-1.0-pro',
        'gemini-pro'  # Old model name for comparison
    ]
    
    successful_models = []
    
    for model_name in models_to_test:
        print(f"\n🧪 Testing model: {model_name}")
        
        try:
            model = genai.GenerativeModel(model_name)
            
            # Simple test prompt
            response = model.generate_content("Hello! Please respond with just 'API test successful'")
            
            if response and response.text:
                print(f"✅ {model_name}: {response.text.strip()}")
                successful_models.append(model_name)
            else:
                print(f"❌ {model_name}: Empty response")
                
        except Exception as e:
            print(f"❌ {model_name}: {str(e)[:100]}...")
    
    print(f"\n📊 Results:")
    print(f"✅ Working models: {len(successful_models)}")
    print(f"❌ Failed models: {len(models_to_test) - len(successful_models)}")
    
    if successful_models:
        print(f"🎯 Recommended model: {successful_models[0]}")
        return successful_models[0]
    else:
        print("❌ No models worked with your API key")
        return False

def test_brewery_decision(model_name):
    """Test the model with a brewery decision prompt"""
    
    print(f"\n🍺 Testing brewery decision-making with {model_name}...")
    
    try:
        model = genai.GenerativeModel(model_name)
        
        brewery_prompt = """
You are an AI making brewery business decisions.

Current Situation:
- Inventory: 85 units
- Current Price: $10.50
- Competitor Prices: [$9.20, $11.80, $10.00]
- Turn: 5

Make strategic decisions for:
- Price (range: $8.00-$15.00)
- Production (range: 10-150 units)
- Marketing spend (range: $0-$2000)

Respond in JSON format:
{"price": 10.75, "production": 60, "marketing": 800, "reasoning": "Your strategy explanation"}
"""
        
        response = model.generate_content(brewery_prompt)
        
        if response and response.text:
            print(f"🤖 AI Response:")
            print(response.text)
            
            # Try to parse as JSON
            try:
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    print(f"\n✅ Parsed JSON successfully:")
                    print(f"   Price: ${data.get('price', 'N/A')}")
                    print(f"   Production: {data.get('production', 'N/A')} units")
                    print(f"   Marketing: ${data.get('marketing', 'N/A')}")
                    print(f"   Reasoning: {data.get('reasoning', 'N/A')[:100]}...")
                    return True
                else:
                    print("⚠️ No JSON found in response")
                    return False
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse JSON: {e}")
                return False
        else:
            print("❌ Empty response from model")
            return False
            
    except Exception as e:
        print(f"❌ Brewery test failed: {e}")
        return False

def list_available_models():
    """List all available models"""
    print("\n📋 Listing available models...")
    
    try:
        models = genai.list_models()
        print("Available models:")
        for model in models:
            print(f"  - {model.name}")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

if __name__ == "__main__":
    print("🧪 Gemini API Key Test Script")
    print("=" * 40)
    
    # Test API key and find working model
    working_model = test_gemini_api_key()
    
    if working_model:
        print(f"\n🎯 Your API key works! Best model: {working_model}")
        
        # Test brewery-specific functionality
        brewery_success = test_brewery_decision(working_model)
        
        if brewery_success:
            print("\n🎉 SUCCESS! Your API key is fully compatible with BrewMasters!")
            print(f"💡 Use this model in your game: {working_model}")
        else:
            print("\n⚠️ API key works but brewery decision format needs adjustment")
    else:
        print("\n❌ API key test failed")
        print("\n🔍 Troubleshooting steps:")
        print("1. Check your API key at: https://ai.google.dev/")
        print("2. Make sure billing is enabled if required")
        print("3. Verify API key permissions")
        print("4. Try generating a new API key")
    
    # List available models for reference
    if working_model:
        list_available_models()
    
    print("\n" + "=" * 40)
    print("🔚 Test complete!")