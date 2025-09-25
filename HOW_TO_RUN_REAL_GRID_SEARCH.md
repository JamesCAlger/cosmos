# How to Run Real Grid Search

## Quick Start (Minimal Cost Version)

### 1. Install Dependencies
```bash
pip install openai loguru numpy datasets
```

### 2. Set OpenAI API Key (Optional)
```bash
# Windows Command Prompt
set OPENAI_API_KEY=your-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run Minimal Grid Search
```bash
python scripts/run_minimal_real_grid_search.py
```

This will:
- Test 27 configurations (3x3x3 grid)
- Use 10 documents and 3 test queries
- Cost approximately $0.04 total if using OpenAI
- Work in mock mode if no API key is set

## Cost Breakdown

### Minimal Version (Recommended for Testing)
- **Documents**: 10
- **Queries**: 3
- **Configurations**: 27
- **API Calls**: 81 total (3 queries × 27 configs)
- **Estimated Cost**: ~$0.04

### Full Version
- **Documents**: 100-1000
- **Queries**: 20
- **Configurations**: 27
- **API Calls**: 540+ (20 queries × 27 configs)
- **Estimated Cost**: $2-5

## Three Ways to Run

### Option 1: Mock Mode (No Cost)
Don't set OPENAI_API_KEY. The system will:
- Use real chunking and retrieval
- Return mock answers
- Still demonstrate the grid search architecture

### Option 2: Minimal Real Mode (Low Cost)
Set OPENAI_API_KEY and run `run_minimal_real_grid_search.py`:
- Real OpenAI API calls
- Minimal data (10 docs, 3 queries)
- ~$0.04 total cost

### Option 3: Full Real Mode (Higher Cost)
Set OPENAI_API_KEY and run `run_real_grid_search.py`:
- Full MS MARCO dataset
- Real embeddings and generation
- ~$2-5 total cost

## Understanding the Results

The grid search tests combinations of:
- **Chunk Sizes**: 256, 512, 1024 characters
- **Top-K Retrieval**: 3, 5, 10 documents
- **Temperature**: 0.0, 0.3, 0.7

Output shows:
```
Config | Chunk | TopK | Temp | Accuracy | Latency | Cost
  1/27 |   256 |    3 | 0.0 | 0.850    | 1.23s   | $0.0015
  2/27 |   256 |    3 | 0.3 | 0.867    | 1.19s   | $0.0015
  ...
```

Best configuration will be saved to `minimal_real_results.json`.

## Customization

Edit the search space in the script:
```python
search_space = {
    "chunker.fixed_size.chunk_size": [256, 512, 1024],  # Add more sizes
    "retriever.simple.top_k": [3, 5, 10],               # Add more k values
    "generator.openai_mini.temperature": [0.0, 0.3, 0.7] # Add more temps
}
```

## Monitoring Costs

The script tracks costs in real-time:
- Each query costs ~$0.0005 with GPT-3.5-turbo
- Embeddings are mocked to save costs
- Total cost is shown for each configuration

## Adding Your Own Components

To add a new component, just use the decorator:

```python
@auto_register("retriever", "my_custom")
class MyCustomRetriever(Retriever, SelfDescribingComponent):
    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="MyCustomRetriever",
            type="retriever",
            parameters={
                "my_param": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=[1, 2, 3],
                    tunable=True
                )
            },
            inputs=["query", "chunks"],
            outputs=["retrieved_documents"],
            tunable_params=["my_param"]
        )

    def retrieve(self, query, chunks):
        # Your retrieval logic
        return chunks[:self.config.get("my_param", 2)]
```

Then add it to the search space:
```python
search_space["retriever.my_custom.my_param"] = [1, 2, 3]
```

The system will automatically:
- Register the component
- Include it in the grid search
- Wire it into the pipeline
- Track its parameters

## Troubleshooting

### "No module named 'openai'"
Run: `pip install openai`

### "Invalid API key"
Check your API key is correctly set as an environment variable

### "Rate limit exceeded"
Add delays between API calls or reduce the number of queries

### Mock mode not working
The system automatically falls back to mock mode if no API key is set