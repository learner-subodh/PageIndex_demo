"""
PageIndex Utilities - Hugging Face Edition
Replaces OpenAI API calls with local Hugging Face models
"""

import yaml
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from config.yaml"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'model': {
                'name': 'mistralai/Mistral-7B-Instruct-v0.2',
                'max_tokens': 2048,
                'temperature': 0.1,
                'device': 'auto'
            },
            'document': {
                'toc_check_pages': 20,
                'max_pages_per_node': 10,
                'max_tokens_per_node': 20000
            },
            'tree': {
                'if_add_node_id': True,
                'if_add_node_summary': True,
                'if_add_doc_description': True,
                'if_add_node_text': False
            },
            'output': {
                'format': 'json',
                'indent': 2
            }
        }
    
    def load(self, user_opt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration with user overrides
        
        Args:
            user_opt: User-provided options to override defaults
            
        Returns:
            Merged configuration dictionary
        """
        config = self.config.copy()
        
        if user_opt:
            # Merge user options with config
            if 'model' in user_opt:
                config['model']['name'] = user_opt.get('model', config['model']['name'])
            
            # Tree settings
            config['tree']['if_add_node_id'] = user_opt.get('if_add_node_id', config['tree']['if_add_node_id'])
            config['tree']['if_add_node_summary'] = user_opt.get('if_add_node_summary', config['tree']['if_add_node_summary'])
            config['tree']['if_add_doc_description'] = user_opt.get('if_add_doc_description', config['tree']['if_add_doc_description'])
            config['tree']['if_add_node_text'] = user_opt.get('if_add_node_text', config['tree']['if_add_node_text'])
            
            # Document settings
            if 'toc_check_pages' in user_opt:
                config['document']['toc_check_pages'] = user_opt['toc_check_pages']
            if 'max_pages_per_node' in user_opt:
                config['document']['max_pages_per_node'] = user_opt['max_pages_per_node']
            if 'max_tokens_per_node' in user_opt:
                config['document']['max_tokens_per_node'] = user_opt['max_tokens_per_node']
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class HuggingFaceModel:
    """
    Wrapper for Hugging Face models to replace OpenAI API
    Provides JSON-structured output for document indexing
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 device: str = "auto",
                 max_tokens: int = 2048,
                 temperature: float = 0.1):
        """
        Initialize Hugging Face model
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda', 'cpu', or 'auto')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"Loading model: {model_name}")
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Load model with appropriate settings
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for instruction
            
        Returns:
            Generated text
        """
        # Format prompt based on model type
        if "mistral" in self.model_name.lower() or "zephyr" in self.model_name.lower():
            if system_prompt:
                full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
            else:
                full_prompt = f"<s>[INST] {prompt} [/INST]"
        elif "llama" in self.model_name.lower():
            if system_prompt:
                full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            else:
                full_prompt = f"[INST] {prompt} [/INST]"
        else:
            # Generic format
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]")[-1].strip()
        elif "<</SYS>>" in generated_text:
            parts = generated_text.split("[/INST]")
            if len(parts) > 1:
                generated_text = parts[-1].strip()
        
        return generated_text
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate JSON output from prompt
        
        Args:
            prompt: User prompt requesting JSON output
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON dictionary
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. No additional text before or after the JSON."
        
        if system_prompt:
            json_system = f"{system_prompt}\n\nYou MUST respond with valid JSON only. No explanations, no markdown, just pure JSON."
        else:
            json_system = "You are a helpful assistant that responds ONLY with valid JSON. No additional text."
        
        # Generate text
        response = self.generate(json_prompt, json_system)
        
        # Try to extract and parse JSON
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {response[:200]}...")
            logger.error(f"JSON Error: {e}")
            # Return empty structure
            return {}


def extract_text_from_pdf(pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> str:
    """
    Extract text from PDF pages
    
    Args:
        pdf_path: Path to PDF file
        start_page: Starting page (0-indexed)
        end_page: Ending page (0-indexed, None for all pages)
        
    Returns:
        Extracted text
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(pdf_path)
    
    if end_page is None:
        end_page = len(doc)
    
    text = ""
    for page_num in range(start_page, min(end_page, len(doc))):
        page = doc[page_num]
        text += page.get_text()
    
    doc.close()
    return text


def save_json(data: Dict[str, Any], output_path: str, indent: int = 2):
    """Save dictionary to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(json_path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
