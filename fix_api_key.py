"""Fix API key trailing space issue"""
import os

def fix_api_key():
    # Read from .env
    with open('.env', 'r') as f:
        lines = f.readlines()

    # Find and clean the API key
    for i, line in enumerate(lines):
        if line.startswith('OPENAI_API_KEY='):
            # Extract the key
            key_part = line.split('=', 1)[1]
            # Remove ALL whitespace including newlines and spaces
            clean_key = key_part.strip()

            print(f"Original key length: {len(key_part)}")
            print(f"Cleaned key length: {len(clean_key)}")
            print(f"Key starts with: {clean_key[:20]}")
            print(f"Key ends with: ...{clean_key[-10:]}")

            # Check for issues
            if key_part != clean_key + '\n':
                print("\n⚠ FOUND ISSUE: Extra whitespace detected!")
                print(f"Last char code: {ord(key_part[-1]) if key_part else 'empty'}")
                if len(key_part) > len(clean_key) + 1:
                    print(f"Extra chars: {repr(key_part[len(clean_key):])}")

            # Write back the cleaned version
            lines[i] = f'OPENAI_API_KEY={clean_key}\n'

            with open('.env', 'w') as f:
                f.writelines(lines)

            print("\n✓ .env file has been cleaned")

            # Also set it in environment
            os.environ['OPENAI_API_KEY'] = clean_key

            # Test it
            print("\nTesting cleaned key...")
            test_cleaned_key(clean_key)

            return clean_key

    print("ERROR: OPENAI_API_KEY not found in .env")
    return None

def test_cleaned_key(api_key):
    """Test the cleaned API key"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Simple test
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'working'"}],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"✓ SUCCESS! API responded: {result}")
        return True

    except Exception as e:
        print(f"✗ Still failing: {e}")

        # Check the error details
        if "Illegal header" in str(e):
            print("\nThe key might be corrupted. Please:")
            print("1. Go to https://platform.openai.com/api-keys")
            print("2. Create a new API key")
            print("3. Copy it carefully (no spaces)")
            print("4. Replace it in the .env file")

        return False

if __name__ == "__main__":
    fix_api_key()