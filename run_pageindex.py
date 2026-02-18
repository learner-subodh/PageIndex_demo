#!/usr/bin/env python3
"""
PageIndex Runner - Hugging Face Edition
Main script to run PageIndex on PDF documents using local Hugging Face models
"""

import argparse
import os
import sys
from pathlib import Path
import logging

from utils import ConfigLoader
from page_index import PageIndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='PageIndex: Build hierarchical tree structure from PDF documents using HuggingFace models'
    )
    
    # Required arguments
    parser.add_argument(
        '--pdf_path',
        type=str,
        required=True,
        help='Path to PDF file to process'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mistral-7B-Instruct-v0.2',
        help='HuggingFace model name (default: mistralai/Mistral-7B-Instruct-v0.2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for JSON file (default: <pdf_name>_pageindex.json)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--toc-check-pages',
        type=int,
        default=20,
        help='Number of pages to check for table of contents (default: 20)'
    )
    
    parser.add_argument(
        '--max-pages-per-node',
        type=int,
        default=10,
        help='Maximum pages per node (default: 10)'
    )
    
    parser.add_argument(
        '--max-tokens-per-node',
        type=int,
        default=20000,
        help='Maximum tokens per node (default: 20000)'
    )
    
    parser.add_argument(
        '--if-add-node-id',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='Add node IDs to tree structure (default: yes)'
    )
    
    parser.add_argument(
        '--if-add-node-summary',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='Add node summaries to tree structure (default: yes)'
    )
    
    parser.add_argument(
        '--if-add-doc-description',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='Add document description to tree structure (default: yes)'
    )
    
    parser.add_argument(
        '--if-add-node-text',
        type=str,
        choices=['yes', 'no'],
        default='no',
        help='Add full node text to tree structure (default: no)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to run model on (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Validate PDF path
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        pdf_name = Path(args.pdf_path).stem
        args.output = f"{pdf_name}_pageindex.json"
    
    logger.info("=" * 60)
    logger.info("PageIndex - Hugging Face Edition")
    logger.info("=" * 60)
    logger.info(f"PDF: {args.pdf_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    
    # Create user options
    user_opt = {
        'model': args.model,
        'if_add_node_id': args.if_add_node_id == 'yes',
        'if_add_node_summary': args.if_add_node_summary == 'yes',
        'if_add_doc_description': args.if_add_doc_description == 'yes',
        'if_add_node_text': args.if_add_node_text == 'yes',
        'toc_check_pages': args.toc_check_pages,
        'max_pages_per_node': args.max_pages_per_node,
        'max_tokens_per_node': args.max_tokens_per_node
    }
    
    # Load config with user overrides
    config = config_loader.load(user_opt)
    
    # Override device if specified
    config['model']['device'] = args.device
    
    try:
        # Build PageIndex
        builder = PageIndexBuilder(config)
        tree = builder.build_index(args.pdf_path, args.output)
        
        logger.info("=" * 60)
        logger.info("✅ PageIndex generation completed successfully!")
        logger.info(f"📄 Output saved to: {args.output}")
        logger.info(f"📊 Total pages: {tree.get('total_pages', 'N/A')}")
        logger.info(f"🌲 Total nodes: {len(tree.get('nodes', []))}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error building PageIndex: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
