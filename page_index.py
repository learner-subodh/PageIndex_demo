"""
PageIndex - Main Module (Hugging Face Edition)
Builds hierarchical tree structure from PDF documents using local LLMs
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from utils import (
    HuggingFaceModel, 
    extract_text_from_pdf, 
    save_json,
    ConfigLoader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PageIndexBuilder:
    """
    Build PageIndex tree structure from PDF documents
    Uses Hugging Face models for structure generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PageIndex builder
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize HuggingFace model
        model_config = config.get('model', {})
        self.model = HuggingFaceModel(
            model_name=model_config.get('name', 'mistralai/Mistral-7B-Instruct-v0.2'),
            device=model_config.get('device', 'auto'),
            max_tokens=model_config.get('max_tokens', 2048),
            temperature=model_config.get('temperature', 0.1)
        )
        
        self.tree_config = config.get('tree', {})
        self.doc_config = config.get('document', {})
        
        self.node_counter = 0
    
    def _get_next_node_id(self) -> str:
        """Generate next node ID"""
        node_id = f"{self.node_counter:04d}"
        self.node_counter += 1
        return node_id
    
    def detect_table_of_contents(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Detect if PDF has a table of contents
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            TOC structure if found, None otherwise
        """
        logger.info("Checking for table of contents...")
        
        # Extract first N pages
        toc_pages = self.doc_config.get('toc_check_pages', 20)
        text = extract_text_from_pdf(pdf_path, start_page=0, end_page=toc_pages)
        
        prompt = f"""Analyze the following text from the first {toc_pages} pages of a document.
Determine if there is a table of contents (TOC) present.

If a TOC exists, extract it and return a JSON object with:
{{
    "has_toc": true,
    "toc_start_page": <page number where TOC starts>,
    "toc_end_page": <page number where TOC ends>,
    "entries": [
        {{"title": "Chapter/Section title", "page": page_number}},
        ...
    ]
}}

If no TOC exists, return:
{{
    "has_toc": false
}}

Document text:
{text[:4000]}...

Respond with ONLY valid JSON."""

        try:
            result = self.model.generate_json(prompt)
            
            if result.get('has_toc', False):
                logger.info(f"Found TOC with {len(result.get('entries', []))} entries")
                return result
            else:
                logger.info("No TOC detected")
                return None
        except Exception as e:
            logger.warning(f"Error detecting TOC: {e}")
            return None
    
    def generate_document_description(self, pdf_path: str) -> str:
        """
        Generate high-level description of document
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document description
        """
        if not self.tree_config.get('if_add_doc_description', True):
            return ""
        
        logger.info("Generating document description...")
        
        # Extract first few pages and last page
        text_start = extract_text_from_pdf(pdf_path, start_page=0, end_page=3)
        
        prompt = f"""Analyze this document excerpt and provide a concise description (2-3 sentences) of what this document is about, its purpose, and main topics covered.

Document excerpt:
{text_start[:3000]}

Provide only the description, no additional commentary."""

        system_prompt = "You are a document analyst. Provide clear, concise descriptions."
        
        try:
            description = self.model.generate(prompt, system_prompt)
            logger.info(f"Generated description: {description[:100]}...")
            return description.strip()
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Document description unavailable."
    
    def create_tree_structure(self, pdf_path: str, toc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create hierarchical tree structure from PDF
        
        Args:
            pdf_path: Path to PDF file
            toc: Optional table of contents structure
            
        Returns:
            Tree structure as dictionary
        """
        logger.info("Creating tree structure...")
        
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Generate document description
        doc_description = self.generate_document_description(pdf_path)
        
        if toc and toc.get('has_toc', False):
            # Use TOC to build structure
            tree = self._build_tree_from_toc(pdf_path, toc, total_pages, doc_description)
        else:
            # Build structure by analyzing content
            tree = self._build_tree_from_content(pdf_path, total_pages, doc_description)
        
        return tree
    
    def _build_tree_from_toc(self, pdf_path: str, toc: Dict[str, Any], 
                            total_pages: int, doc_description: str) -> Dict[str, Any]:
        """Build tree structure using table of contents"""
        logger.info("Building tree from TOC...")
        
        entries = toc.get('entries', [])
        nodes = []
        
        for i, entry in enumerate(entries):
            start_page = entry.get('page', i)
            # End page is start of next entry or end of document
            if i < len(entries) - 1:
                end_page = entries[i + 1].get('page', total_pages)
            else:
                end_page = total_pages
            
            node = self._create_node(
                title=entry.get('title', f'Section {i+1}'),
                start_page=start_page,
                end_page=end_page,
                pdf_path=pdf_path
            )
            nodes.append(node)
        
        # Create root structure
        root = {
            "document_description": doc_description,
            "total_pages": total_pages,
            "nodes": nodes
        }
        
        return root
    
    def _build_tree_from_content(self, pdf_path: str, total_pages: int, 
                                doc_description: str) -> Dict[str, Any]:
        """Build tree structure by analyzing content"""
        logger.info("Building tree from content analysis...")
        
        max_pages_per_node = self.doc_config.get('max_pages_per_node', 10)
        nodes = []
        
        current_page = 0
        while current_page < total_pages:
            end_page = min(current_page + max_pages_per_node, total_pages)
            
            # Extract text for this section
            text = extract_text_from_pdf(pdf_path, start_page=current_page, end_page=end_page)
            
            # Generate section title and summary
            node_info = self._analyze_section(text, current_page, end_page)
            
            node = {
                "title": node_info.get('title', f'Pages {current_page+1}-{end_page}'),
                "start_index": current_page,
                "end_index": end_page
            }
            
            if self.tree_config.get('if_add_node_id', True):
                node['node_id'] = self._get_next_node_id()
            
            if self.tree_config.get('if_add_node_summary', True):
                node['summary'] = node_info.get('summary', '')
            
            nodes.append(node)
            current_page = end_page
        
        # Create root structure
        root = {
            "document_description": doc_description,
            "total_pages": total_pages,
            "nodes": nodes
        }
        
        return root
    
    def _analyze_section(self, text: str, start_page: int, end_page: int) -> Dict[str, Any]:
        """
        Analyze a section of text to extract title and summary
        
        Args:
            text: Section text
            start_page: Starting page number
            end_page: Ending page number
            
        Returns:
            Dictionary with title and summary
        """
        # Truncate text if too long
        text_preview = text[:2000] if len(text) > 2000 else text
        
        prompt = f"""Analyze this section from pages {start_page+1} to {end_page} of a document.

Provide:
1. A concise title (5-10 words) that captures the main topic
2. A brief summary (2-3 sentences) of what this section covers

Section text:
{text_preview}

Return ONLY valid JSON in this format:
{{
    "title": "Section Title Here",
    "summary": "Brief summary here..."
}}"""

        try:
            result = self.model.generate_json(prompt)
            return {
                'title': result.get('title', f'Pages {start_page+1}-{end_page}'),
                'summary': result.get('summary', 'Section summary unavailable.')
            }
        except Exception as e:
            logger.warning(f"Error analyzing section: {e}")
            return {
                'title': f'Pages {start_page+1}-{end_page}',
                'summary': 'Section summary unavailable.'
            }
    
    def _create_node(self, title: str, start_page: int, end_page: int, 
                    pdf_path: str) -> Dict[str, Any]:
        """Create a tree node with metadata"""
        node = {
            "title": title,
            "start_index": start_page,
            "end_index": end_page
        }
        
        if self.tree_config.get('if_add_node_id', True):
            node['node_id'] = self._get_next_node_id()
        
        if self.tree_config.get('if_add_node_summary', True):
            # Extract text and generate summary
            text = extract_text_from_pdf(pdf_path, start_page=start_page, end_page=end_page)
            summary = self._generate_summary(text)
            node['summary'] = summary
        
        return node
    
    def _generate_summary(self, text: str) -> str:
        """Generate summary for text"""
        text_preview = text[:2000] if len(text) > 2000 else text
        
        prompt = f"""Provide a concise summary (2-3 sentences) of the following text:

{text_preview}

Summary:"""

        system_prompt = "You are a summarization expert. Provide clear, concise summaries."
        
        try:
            summary = self.model.generate(prompt, system_prompt)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary unavailable."
    
    def build_index(self, pdf_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Build complete PageIndex for PDF
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional path to save output JSON
            
        Returns:
            Complete tree structure
        """
        logger.info(f"Building PageIndex for: {pdf_path}")
        
        # Check for TOC
        toc = self.detect_table_of_contents(pdf_path)
        
        # Create tree structure
        tree = self.create_tree_structure(pdf_path, toc)
        
        # Save if output path provided
        if output_path:
            save_json(tree, output_path, indent=self.config.get('output', {}).get('indent', 2))
            logger.info(f"Saved PageIndex to: {output_path}")
        
        logger.info("PageIndex generation complete!")
        return tree


def build_pageindex(pdf_path: str, config: Optional[Dict[str, Any]] = None, 
                   output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to build PageIndex
    
    Args:
        pdf_path: Path to PDF file
        config: Optional configuration dictionary
        output_path: Optional path to save output
        
    Returns:
        Tree structure
    """
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.load()
    
    builder = PageIndexBuilder(config)
    return builder.build_index(pdf_path, output_path)
