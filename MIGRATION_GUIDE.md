# Migration Guide: OpenAI → HuggingFace

## Overview of Changes

This modified version of PageIndex replaces OpenAI API calls with local HuggingFace models. Here's what changed and why:

## Key Changes

### 1. Model Wrapper (`utils.py`)

**Original (OpenAI):**
```python
import openai

openai.api_key = os.getenv("CHATGPT_API_KEY")
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

**Modified (HuggingFace):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0])
```

### 2. Configuration

**Original:**
- Requires `.env` file with API key
- Model selection limited to OpenAI models

**Modified:**
- No API key needed
- `config.yaml` for configuration
- Any HuggingFace model can be used

### 3. Dependencies

**Original Requirements:**
```
openai>=1.0.0
python-dotenv>=0.19.0
pymupdf>=1.24.0
```

**Modified Requirements:**
```
transformers>=4.36.0
torch>=2.0.0
accelerate>=0.25.0
pymupdf>=1.24.0
```

## Feature Comparison

| Feature | Original | HF Version |
|---------|----------|------------|
| **Cost** | Pay per API call | Free |
| **Privacy** | Data sent to OpenAI | Runs locally |
| **Speed** | Fast (API) | Slower (local inference) |
| **Quality** | GPT-4 quality | Good (depends on model) |
| **Setup** | Easy (API key) | Moderate (model download) |
| **Hardware** | Any | GPU recommended |
| **Internet** | Required | Not required (after setup) |

## Model Recommendations

### For Best Quality (if you have GPU)
```bash
--model "mistralai/Mistral-7B-Instruct-v0.2"
```

### For Faster Processing (smaller model)
```bash
--model "google/flan-t5-large"
```

### For Best JSON Structure
```bash
--model "HuggingFaceH4/zephyr-7b-beta"
```

## Usage Comparison

### Original PageIndex

```bash
# Setup
echo "CHATGPT_API_KEY=sk-..." > .env

# Run
python3 run_pageindex.py --pdf_path document.pdf --model gpt-4o-2024-11-20
```

### HuggingFace Version

```bash
# Setup (one time)
pip install -r requirements.txt

# Run (no API key needed)
python run_pageindex.py --pdf_path document.pdf
```

## Code Integration

### Original

```python
from pageindex import PageIndex

# Requires OpenAI API key in environment
pageindex = PageIndex()
tree = pageindex.build(pdf_path="document.pdf")
```

### Modified

```python
from page_index import build_pageindex

# No API key needed
tree = build_pageindex(pdf_path="document.pdf")
```

## Performance Expectations

### Original (OpenAI API)
- **Speed**: 1-3 minutes for typical document
- **Cost**: $0.01-0.10 per document (varies by model)
- **Quality**: Excellent (GPT-4)

### Modified (HuggingFace)
- **Speed**: 
  - With GPU: 3-10 minutes
  - With CPU: 15-30 minutes
- **Cost**: Free (one-time model download ~14GB)
- **Quality**: Very good (depends on model choice)

## When to Use Each Version

### Use Original (OpenAI) If:
- ✅ You need the absolute best quality
- ✅ You want fastest processing
- ✅ You don't mind API costs
- ✅ You're processing documents infrequently

### Use Modified (HuggingFace) If:
- ✅ You want zero API costs
- ✅ You need privacy (local processing)
- ✅ You're processing many documents
- ✅ You have access to GPU
- ✅ You don't have OpenAI API access

## Common Issues & Solutions

### Issue 1: Out of Memory
**Solution**: Use smaller model or CPU
```bash
--model "google/flan-t5-large" --device cpu
```

### Issue 2: Slow Processing
**Solution**: Use GPU or reduce pages per node
```bash
--device cuda --max-pages-per-node 5
```

### Issue 3: Poor JSON Quality
**Solution**: Use better instruction-following model
```bash
--model "mistralai/Mistral-7B-Instruct-v0.2"
```

### Issue 4: Model Download Fails
**Solution**: Download manually first
```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
```

## Output Format

**Good news**: The output JSON format is **identical** between versions!

Both produce:
```json
{
  "document_description": "...",
  "total_pages": 50,
  "nodes": [
    {
      "title": "...",
      "node_id": "0001",
      "start_index": 0,
      "end_index": 5,
      "summary": "..."
    }
  ]
}
```

## Transitioning Existing Projects

If you're using the original PageIndex:

1. **Install new dependencies**:
   ```bash
   pip install transformers torch accelerate
   ```

2. **Replace import**:
   ```python
   # Old
   from pageindex import PageIndex
   
   # New
   from page_index import build_pageindex
   ```

3. **Update code**:
   ```python
   # Old
   pageindex = PageIndex()
   tree = pageindex.build(pdf_path="doc.pdf")
   
   # New
   tree = build_pageindex(pdf_path="doc.pdf")
   ```

4. **Remove API key**: Delete `.env` file (no longer needed)

## Hybrid Approach

You can also keep both versions and switch based on needs:

```python
import os

if os.getenv("USE_OPENAI") == "true":
    from pageindex_original import PageIndex
    builder = PageIndex()
else:
    from page_index import build_pageindex
    builder = build_pageindex

tree = builder(pdf_path="document.pdf")
```

## Advanced: Custom Model Integration

Want to use a specific model? Easy!

```python
from utils import HuggingFaceModel, ConfigLoader
from page_index import PageIndexBuilder

# Load config
config = ConfigLoader().load()

# Override model
config['model']['name'] = "your-org/your-model"

# Build
builder = PageIndexBuilder(config)
tree = builder.build_index("document.pdf")
```

## Support Matrix

| Feature | Original | HF Version |
|---------|----------|------------|
| PDF Processing | ✅ | ✅ |
| TOC Detection | ✅ | ✅ |
| Node Summaries | ✅ | ✅ |
| Document Description | ✅ | ✅ |
| Markdown Support | ✅ | ⚠️ (basic) |
| Vision RAG | ✅ | 🔄 (coming soon) |
| Streaming | ✅ | ❌ |
| Batch Processing | ✅ | ✅ |

## Conclusion

The HuggingFace version provides a **free, privacy-focused alternative** to the original PageIndex while maintaining compatibility with the output format. Choose based on your specific needs for cost, speed, and privacy.
