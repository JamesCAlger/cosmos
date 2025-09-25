"""Diagnose OpenAI connection issues"""
import os
import socket
import requests
import ssl
import urllib.request
from urllib.parse import urlparse

def test_basic_internet():
    """Test basic internet connectivity"""
    print("1. Testing basic internet connectivity...")
    try:
        response = requests.get("https://www.google.com", timeout=5)
        print("   ✓ Internet connection OK")
        return True
    except Exception as e:
        print(f"   ✗ No internet: {e}")
        return False

def test_openai_endpoint():
    """Test if OpenAI endpoints are reachable"""
    print("\n2. Testing OpenAI API endpoint reachability...")
    endpoints = [
        "https://api.openai.com",
        "https://api.openai.com/v1/models"
    ]

    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5, headers={
                "User-Agent": "Mozilla/5.0"
            })
            print(f"   ✓ {endpoint}: Status {response.status_code}")
        except requests.exceptions.SSLError as e:
            print(f"   ✗ SSL Error for {endpoint}: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"   ✗ Connection Error for {endpoint}: {e}")
        except Exception as e:
            print(f"   ✗ Error for {endpoint}: {type(e).__name__}: {e}")

def test_dns_resolution():
    """Test DNS resolution for OpenAI"""
    print("\n3. Testing DNS resolution...")
    try:
        ip = socket.gethostbyname("api.openai.com")
        print(f"   ✓ api.openai.com resolves to: {ip}")
    except Exception as e:
        print(f"   ✗ DNS resolution failed: {e}")

def test_ssl_certificate():
    """Test SSL certificate verification"""
    print("\n4. Testing SSL certificate...")
    try:
        import certifi
        print(f"   Using certifi bundle: {certifi.where()}")
    except ImportError:
        print("   ⚠ certifi not installed")

    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen("https://api.openai.com", context=ctx) as response:
            print("   ✓ SSL certificate OK")
    except Exception as e:
        print(f"   ✗ SSL verification failed: {e}")

def check_proxy_settings():
    """Check for proxy configurations"""
    print("\n5. Checking proxy settings...")
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    has_proxy = False

    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"   Found {var}: {value}")
            has_proxy = True

    if not has_proxy:
        print("   No proxy settings found")
    else:
        print("   ⚠ Proxy detected - this might interfere with OpenAI API")

def test_with_curl():
    """Test using system curl command"""
    print("\n6. Testing with curl command...")
    import subprocess

    try:
        # Test basic connectivity
        result = subprocess.run(
            ["curl", "-I", "https://api.openai.com"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("   ✓ curl can reach api.openai.com")
            print(f"   Headers: {result.stdout[:200]}...")
        else:
            print(f"   ✗ curl failed: {result.stderr}")
    except FileNotFoundError:
        print("   ⚠ curl not found (this is OK on Windows)")
    except Exception as e:
        print(f"   ✗ Error: {e}")

def test_openai_sdk_directly():
    """Test OpenAI SDK with detailed error info"""
    print("\n7. Testing OpenAI Python SDK...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ✗ No OPENAI_API_KEY found")
        return

    try:
        from openai import OpenAI

        # Try with explicit timeout
        client = OpenAI(
            api_key=api_key,
            timeout=30.0,  # 30 second timeout
            max_retries=0  # No retries for testing
        )

        # Try to list models (simpler than chat completion)
        print("   Attempting to list models...")
        models = client.models.list()
        print(f"   ✓ Success! Found {len(list(models))} models")

    except Exception as e:
        print(f"   ✗ SDK Error: {type(e).__name__}")
        print(f"   Details: {str(e)}")

        # Check for specific error types
        import traceback
        print("\n   Full traceback:")
        traceback.print_exc()

def check_windows_defender():
    """Check Windows Defender/Firewall status"""
    print("\n8. Checking Windows Defender Firewall...")
    try:
        import subprocess
        result = subprocess.run(
            ["netsh", "advfirewall", "show", "currentprofile"],
            capture_output=True,
            text=True
        )
        if "State" in result.stdout:
            if "ON" in result.stdout:
                print("   ⚠ Windows Firewall is ON (might block connections)")
                print("   Try adding Python to firewall exceptions")
            else:
                print("   ✓ Windows Firewall is OFF")
    except Exception as e:
        print(f"   Could not check firewall: {e}")

def main():
    print("=" * 60)
    print("OpenAI Connection Diagnostic Tool")
    print("=" * 60)

    test_basic_internet()
    test_openai_endpoint()
    test_dns_resolution()
    test_ssl_certificate()
    check_proxy_settings()
    test_with_curl()
    test_openai_sdk_directly()
    check_windows_defender()

    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()