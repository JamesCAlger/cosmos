"""Quick script to check MS MARCO dataset structure"""

from datasets import load_dataset

# Load a small sample to check structure
dataset = load_dataset("ms_marco", "v2.1", split="train", streaming=True)

# Check first few samples
for i, sample in enumerate(dataset):
    if i < 2:
        print(f"\nSample {i+1}:")
        print("Keys:", sample.keys())
        for key, value in sample.items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"  {key}: list of {len(value)} items")
            elif isinstance(value, str):
                print(f"  {key}: '{value[:100]}...'" if len(value) > 100 else f"  {key}: '{value}'")
            else:
                print(f"  {key}: {value}")
    else:
        break