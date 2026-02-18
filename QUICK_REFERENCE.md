# PageIndex HuggingFace - Quick Reference

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
python run_pageindex.py --pdf_path your_document.pdf

# 3. Done! Output: your_document_pageindex.json
```

## 📋 Common Commands

### Basic Usage
```bash
python run_pageindex.py --pdf_path document.pdf
```

### Use Specific Model
```bash
python run_pageindex.py --pdf_path document.pdf \
  --model "HuggingFaceH4/zephyr-7b-beta"
```

### CPU Only
```bash
python run_pageindex.py --pdf_path document.pdf --device cpu
```

### Fast Mode (no summaries)
```bash
python run_pageindex.py --pdf_path document.pdf \
  --if-add-node-summary no --if-add-doc-description no
```

### Custom Output
```bash
python run_pageindex.py --pdf_path document.pdf \
  --output my_custom_tree.json
```

## 🤖 Recommended Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Mistral-7B-Instruct-v0.2 | 14GB | Medium | High | **Default, best balance** |
| zephyr-7b-beta | 14GB | Medium | High | Best for JSON |
| flan-t5-large | 3GB | Fast | Good | Faster, smaller |
| Llama-2-7b-chat | 14GB | Medium | High | Alternative (needs token) |

## ⚙️ Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pdf_path` | Required | PDF file to process |
| `--model` | Mistral-7B | HuggingFace model name |
| `--device` | auto | cuda/cpu/auto |
| `--output` | auto | Output JSON path |
| `--max-pages-per-node` | 10 | Pages per section |
| `--if-add-node-summary` | yes | Add summaries |

## 🐍 Python API

```python
from page_index import build_pageindex

# Simple usage
tree = build_pageindex("document.pdf")

# With custom config
from utils import ConfigLoader
config = ConfigLoader().load({'model': 'zephyr-7b-beta'})
tree = build_pageindex("document.pdf", config=config)

# Access results
print(f"Pages: {tree['total_pages']}")
print(f"Nodes: {len(tree['nodes'])}")
for node in tree['nodes']:
    print(f"- {node['title']}: pages {node['start_index']}-{node['end_index']}")
```

## 📊 Output Format

```json
{
  "document_description": "Brief overview...",
  "total_pages": 50,
  "nodes": [
    {
      "title": "Introduction",
      "node_id": "0001",
      "start_index": 0,
      "end_index": 5,
      "summary": "This section covers..."
    }
  ]
}
```

## 🔧 Troubleshooting

### Out of Memory?
```bash
--device cpu --model "google/flan-t5-large"
```

### Too Slow?
```bash
--if-add-node-summary no --max-pages-per-node 15
```

### Bad JSON Output?
```bash
--model "mistralai/Mistral-7B-Instruct-v0.2"
```

## 💻 Google Colab

```python
# Install
!pip install pymupdf pyyaml transformers torch accelerate

# Upload files
from google.colab import files
uploaded = files.upload()

# Run
!python run_pageindex.py --pdf_path {list(uploaded.keys())[0]}

# Download result
files.download("output_pageindex.json")
```

## 📁 File Structure

```
pageindex_hf/
├── config.yaml                      # Configuration
├── utils.py                         # HF model wrapper
├── page_index.py                    # Main logic
├── run_pageindex.py                 # CLI interface
├── requirements.txt                 # Dependencies
├── README.md                        # Full documentation
├── MIGRATION_GUIDE.md              # OpenAI → HF guide
├── pageindex_huggingface_colab.ipynb  # Colab notebook
└── setup.sh                         # Setup script
```

## 🎯 Key Features

- ✅ No API keys needed
- ✅ Runs completely locally
- ✅ Free and open source
- ✅ GPU acceleration support
- ✅ Multiple model options
- ✅ Compatible with original PageIndex output

## 📞 Getting Help

- **Documentation**: See README.md
- **Migration**: See MIGRATION_GUIDE.md
- **Colab**: Use pageindex_huggingface_colab.ipynb
- **Original Repo**: https://github.com/VectifyAI/PageIndex

## ⚡ Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Smaller models**: Faster but lower quality
3. **Disable summaries**: Skip for speed
4. **Batch process**: Loop through multiple PDFs
5. **Pre-download models**: Avoid delays on first run

```python
# Pre-download model (one time)
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)
```

---

**Questions?** Check README.md for detailed documentation!
