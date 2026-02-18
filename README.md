# PageIndex - Hugging Face Edition

A modified version of [PageIndex](https://github.com/VectifyAI/PageIndex) that uses **free Hugging Face models** instead of OpenAI API for building hierarchical tree structures from PDF documents.

## 🎯 Overview

This implementation replaces OpenAI API calls with local Hugging Face models, enabling you to:
- Build document tree structures **without** requiring an OpenAI API key
- Use **free, open-source** language models
- Run **completely locally** on your machine or Google Colab
- Generate JSON-structured document indexes for RAG systems

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python run_pageindex.py --pdf_path /path/to/your/document.pdf
```

This will:
1. Load the Mistral-7B model (default)
2. Analyze your PDF
3. Generate a hierarchical tree structure
4. Save output to `<pdf_name>_pageindex.json`

## 🤖 Supported Models

Recommended free models optimized for JSON structure generation:

### Primary Recommendation
- **`mistralai/Mistral-7B-Instruct-v0.2`** (default)
  - Best balance of quality and speed
  - Good at following structured output instructions
  - ~7B parameters

### Alternative Models
- **`HuggingFaceH4/zephyr-7b-beta`**
  - Excellent for structured output
  - Good instruction following
  
- **`google/flan-t5-large`**
  - Smaller and faster
  - Good for simpler documents
  
- **`meta-llama/Llama-2-7b-chat-hf`**
  - High quality outputs
  - Requires HuggingFace token for access

## ⚙️ Configuration

### Using Command Line Arguments

```bash
python run_pageindex.py \
    --pdf_path document.pdf \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --output my_index.json \
    --device cuda \
    --max-pages-per-node 10
```

### Using config.yaml

Edit `config.yaml` to set default values:

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
  max_tokens: 2048
  temperature: 0.1
  device: "auto"  # "cuda", "cpu", or "auto"

document:
  toc_check_pages: 20
  max_pages_per_node: 10
  max_tokens_per_node: 20000

tree:
  if_add_node_id: true
  if_add_node_summary: true
  if_add_doc_description: true
  if_add_node_text: false
```

## 📋 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pdf_path` | Path to PDF file (required) | - |
| `--model` | HuggingFace model name | `mistralai/Mistral-7B-Instruct-v0.2` |
| `--output` | Output JSON path | `<pdf_name>_pageindex.json` |
| `--device` | Device: auto/cuda/cpu | `auto` |
| `--toc-check-pages` | Pages to check for TOC | 20 |
| `--max-pages-per-node` | Max pages per node | 10 |
| `--if-add-node-id` | Add node IDs (yes/no) | `yes` |
| `--if-add-node-summary` | Add summaries (yes/no) | `yes` |
| `--if-add-doc-description` | Add doc description (yes/no) | `yes` |

## 🔬 Usage Examples

### Example 1: Basic Usage
```bash
python run_pageindex.py --pdf_path report.pdf
```

### Example 2: Use Different Model
```bash
python run_pageindex.py \
    --pdf_path report.pdf \
    --model "HuggingFaceH4/zephyr-7b-beta"
```

### Example 3: CPU-only Mode
```bash
python run_pageindex.py \
    --pdf_path report.pdf \
    --device cpu
```

### Example 4: Minimal Output (no summaries)
```bash
python run_pageindex.py \
    --pdf_path report.pdf \
    --if-add-node-summary no \
    --if-add-doc-description no
```

## 📊 Output Format

The generated JSON follows the PageIndex format:

```json
{
  "document_description": "This document covers...",
  "total_pages": 50,
  "nodes": [
    {
      "title": "Introduction",
      "node_id": "0001",
      "start_index": 0,
      "end_index": 5,
      "summary": "This section introduces..."
    },
    {
      "title": "Methodology",
      "node_id": "0002",
      "start_index": 5,
      "end_index": 15,
      "summary": "The methodology section describes..."
    }
  ]
}
```

## 💻 Google Colab Usage

### Option 1: Run on Colab with GPU

```python
# Install dependencies
!pip install pymupdf pyyaml transformers torch accelerate

# Clone this repository
!git clone https://github.com/your-repo/pageindex-hf.git
%cd pageindex-hf

# Upload your PDF or use a URL
from google.colab import files
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

# Run PageIndex
!python run_pageindex.py --pdf_path {pdf_path} --device cuda

# Download result
from google.colab import files
files.download(f"{pdf_path.replace('.pdf', '')}_pageindex.json")
```

### Option 2: Use as Python Module

```python
from page_index import build_pageindex

# Build index
tree = build_pageindex(
    pdf_path="document.pdf",
    output_path="output.json"
)

# Use the tree structure
print(f"Total pages: {tree['total_pages']}")
print(f"Nodes: {len(tree['nodes'])}")
```

## 🔧 Key Differences from Original PageIndex

| Feature | Original | This Version |
|---------|----------|--------------|
| LLM Provider | OpenAI API | HuggingFace (local) |
| API Key Required | ✅ Yes | ❌ No |
| Cost | Pay per token | 🆓 Free |
| Models | GPT-4, GPT-3.5 | Mistral, Llama, Flan-T5, etc. |
| Internet Required | ✅ Yes | ❌ No (after download) |
| Hardware | Any | GPU recommended |

## ⚡ Performance Tips

1. **GPU vs CPU**: Use GPU for 10-100x speedup
   ```bash
   --device cuda  # Use GPU
   --device cpu   # Use CPU only
   ```

2. **Smaller Models**: For faster processing with less RAM
   ```bash
   --model "google/flan-t5-large"
   ```

3. **Reduce Node Summaries**: Skip summaries for speed
   ```bash
   --if-add-node-summary no
   ```

4. **Batch Processing**: Process multiple PDFs
   ```bash
   for pdf in *.pdf; do
       python run_pageindex.py --pdf_path "$pdf"
   done
   ```

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Use a smaller model or reduce batch size
```bash
--model "google/flan-t5-large"  # Smaller model
--device cpu  # Use CPU with more RAM
```

### Issue: Model Download Fails
**Solution**: Pre-download models
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)
```

### Issue: JSON Parsing Errors
**Solution**: This can happen with smaller models. Try:
- Using a larger model like Mistral-7B
- Increasing temperature slightly (0.1 → 0.2)
- The code has fallback handling for JSON errors

## 📝 Code Structure

```
pageindex_hf/
├── config.yaml          # Configuration file
├── utils.py            # Utility functions (HF model wrapper)
├── page_index.py       # Main PageIndex logic
├── run_pageindex.py    # Command-line interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔗 Integration with RAG Systems

Use the generated tree structure in your RAG pipeline:

```python
import json
from page_index import build_pageindex

# Build index
tree = build_pageindex("document.pdf")

# Use in RAG
def retrieve_context(query, tree):
    """Simple retrieval example"""
    relevant_nodes = []
    
    for node in tree['nodes']:
        # Simple keyword matching (replace with your logic)
        if any(keyword in node['summary'].lower() 
               for keyword in query.lower().split()):
            relevant_nodes.append({
                'title': node['title'],
                'pages': f"{node['start_index']}-{node['end_index']}",
                'summary': node['summary']
            })
    
    return relevant_nodes

# Example query
results = retrieve_context("financial results", tree)
```

## 📚 Resources

- [Original PageIndex Repository](https://github.com/VectifyAI/PageIndex)
- [PageIndex Documentation](https://docs.pageindex.ai)
- [HuggingFace Models](https://huggingface.co/models)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 🙏 Credits

Based on [PageIndex](https://github.com/VectifyAI/PageIndex) by VectifyAI.
Modified to use HuggingFace models instead of OpenAI API.

## 📄 License

MIT License - See original PageIndex repository for details.
