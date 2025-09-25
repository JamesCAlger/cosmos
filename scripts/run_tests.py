"""Run Week 2 tests for modular architecture"""

import sys
from pathlib import Path
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def run_tests():
    """Run all tests for Week 2 implementation"""
    print("\n" + "="*60)
    print("RUNNING WEEK 2 TESTS")
    print("="*60)

    # Run unit tests
    print("\n1. Running Unit Tests")
    print("-" * 40)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit", "-v"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    # Run integration tests
    print("\n2. Running Integration Tests")
    print("-" * 40)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration", "-v"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    print("\n" + "="*60)
    print("TEST RUN COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_tests()