"""Test OpenAI API connection"""
import os
import sys
from openai import OpenAI

def test_openai_connection():
    """Test if OpenAI API is working"""

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: No OPENAI_API_KEY environment variable found")
        return False

    print(f"Testing with API key: {api_key[:20]}...")

    try:
        # Create client
        client = OpenAI(api_key=api_key)

        # Simple test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello'"}],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"SUCCESS: API responded with: {result}")
        return True

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

        # Provide specific guidance
        error_str = str(e).lower()
        if "api_key" in error_str or "authentication" in error_str:
            print("\nSOLUTION: Your API key appears to be invalid. Please:")
            print("1. Check your .env file has the correct key")
            print("2. Make sure the key starts with 'sk-'")
            print("3. Verify the key hasn't expired")
        elif "connection" in error_str:
            print("\nSOLUTION: Connection issue. Please check:")
            print("1. Your internet connection")
            print("2. Any firewall or VPN blocking OpenAI")
            print("3. Corporate proxy settings")
        elif "rate" in error_str:
            print("\nSOLUTION: Rate limit hit. Please:")
            print("1. Wait a few minutes and try again")
            print("2. Check your OpenAI usage dashboard")

        return False

if __name__ == "__main__":
    success = test_openai_connection()
    sys.exit(0 if success else 1)