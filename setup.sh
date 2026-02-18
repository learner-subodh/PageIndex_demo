#!/bin/bash
# Setup script for PageIndex HuggingFace Edition

echo "=========================================="
echo "PageIndex - HuggingFace Edition Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  python run_pageindex.py --pdf_path your_document.pdf"
echo ""
echo "For GPU support, make sure you have:"
echo "  - CUDA installed"
echo "  - PyTorch with CUDA support"
echo ""
echo "To use with Colab, upload the notebook:"
echo "  pageindex_huggingface_colab.ipynb"
echo ""
