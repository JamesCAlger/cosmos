"""Debug script to test OpenAI API connection issues progressively"""

import sys
import os
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()

import time
import openai
from loguru import logger

# Configure detailed logging
logger.add("connection_debug.log", rotation="10 MB", level="DEBUG")

def test_progressive_batches():
    """Test API calls with increasing batch sizes to find breaking point"""

    print("\n" + "="*80)
    print("PROGRESSIVE BATCH CONNECTION TEST")
    print("="*80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key found!")
        return

    print(f"API Key: {api_key[:10]}...")

    # Test different batch sizes
    batch_sizes = [1, 2, 3, 5, 10, 15, 20, 27]
    delays = [0.15, 0.5, 1.0, 2.0]

    for delay in delays:
        print(f"\n\nTesting with {delay}s delay between calls...")
        print("-"*60)

        for batch_size in batch_sizes:
            print(f"\n  Testing batch size: {batch_size}")

            # Create a single client
            client = openai.OpenAI(api_key=api_key, max_retries=1, timeout=10.0)

            success_count = 0
            error_count = 0
            error_types = {}

            start_time = time.time()

            for i in range(batch_size):
                try:
                    # Wait between calls
                    if i > 0:
                        time.sleep(delay)

                    call_start = time.time()
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Test call {i+1}"}],
                        max_tokens=5,
                        temperature=0.3
                    )
                    call_time = time.time() - call_start

                    success_count += 1
                    print(f"    Call {i+1}/{batch_size}: SUCCESS ({call_time:.2f}s)")

                except Exception as e:
                    error_count += 1
                    error_type = type(e).__name__
                    error_types[error_type] = error_types.get(error_type, 0) + 1

                    print(f"    Call {i+1}/{batch_size}: FAILED - {error_type}: {str(e)[:50]}")
                    logger.error(f"Call {i+1} failed: {e}")

                    # Stop if we hit errors
                    if error_count >= 3:
                        print(f"    Stopping after {error_count} errors...")
                        break

            total_time = time.time() - start_time

            print(f"\n  Results for batch {batch_size}:")
            print(f"    Success: {success_count}/{batch_size}")
            print(f"    Errors: {error_count}")
            if error_types:
                print(f"    Error types: {error_types}")
            print(f"    Total time: {total_time:.2f}s")
            print(f"    Success rate: {success_count/max(1, success_count+error_count)*100:.1f}%")

            # Stop testing larger batches if this one failed
            if error_count > 0:
                print(f"\n  Stopping at batch size {batch_size} due to errors")
                break

            # Small delay between batch tests
            time.sleep(2)


def test_connection_patterns():
    """Test different connection patterns to identify the issue"""

    print("\n" + "="*80)
    print("CONNECTION PATTERN TEST")
    print("="*80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key found!")
        return

    # Test 1: Single client, multiple calls
    print("\n1. Testing single client, multiple rapid calls...")
    client = openai.OpenAI(api_key=api_key, max_retries=0, timeout=5.0)

    for i in range(5):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            elapsed = time.time() - start
            print(f"  Call {i+1}: SUCCESS ({elapsed:.2f}s)")
            time.sleep(0.15)  # Minimal delay
        except Exception as e:
            print(f"  Call {i+1}: FAILED - {type(e).__name__}")
            break

    # Test 2: Multiple clients
    print("\n2. Testing multiple clients (new client each call)...")

    for i in range(5):
        try:
            client = openai.OpenAI(api_key=api_key, max_retries=0, timeout=5.0)
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            elapsed = time.time() - start
            print(f"  Call {i+1}: SUCCESS ({elapsed:.2f}s)")
            time.sleep(0.15)
        except Exception as e:
            print(f"  Call {i+1}: FAILED - {type(e).__name__}")
            break

    # Test 3: Burst pattern (rapid calls, then pause)
    print("\n3. Testing burst pattern (3 rapid, pause, 3 rapid)...")
    client = openai.OpenAI(api_key=api_key, max_retries=0, timeout=5.0)

    for burst in range(2):
        print(f"\n  Burst {burst+1}:")
        for i in range(3):
            try:
                start = time.time()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5
                )
                elapsed = time.time() - start
                print(f"    Call {i+1}: SUCCESS ({elapsed:.2f}s)")
                if i < 2:
                    time.sleep(0.05)  # Very short delay within burst
            except Exception as e:
                print(f"    Call {i+1}: FAILED - {type(e).__name__}")
                break

        if burst == 0:
            print("  Pausing 3 seconds...")
            time.sleep(3)


if __name__ == "__main__":
    print("\nStarting connection diagnostics...")
    print("This will help identify the exact point where connections fail.\n")

    # Run tests
    test_connection_patterns()
    print("\n" + "="*80 + "\n")
    test_progressive_batches()

    print("\n\nDiagnostics complete! Check connection_debug.log for details.")