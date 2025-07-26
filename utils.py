"""
Adobe Round 1A: PDF Text Extraction and Font Analysis
Core utility functions for PDF processing with PyMuPDF
FIXED: Configuration path corrections
"""

import fitz  # PyMuPDF
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return get_default_config()


def get_default_config() -> Dict:
    """Default configuration if config.yaml is missing"""
    return {
        'input_folder': 'input',
        'output_folder': 'output',
        'processing_timeout': 8,
        'hierarchy': {
            'size_ratio_h1': 1.5,
            'size_ratio_h2': 1.25,
            'size_ratio_h3': 1.1,
            'min_heading_length': 3,
            'max_heading_length': 200,
            'title_max_words': 15
        },
        'features': {
            'colon_bonus': 1.25,
            'number_pattern_bonus': 1.2,
            'size_ratio_h1': 1.5,  # Added for backward compatibility
            'size_ratio_h2': 1.25,
            'size_ratio_h3': 1.1
        },
        'model': {
            'name': 'distilbert-base-uncased',
            'max_length': 256
        }
    }


def decode_font_flags(flags: int) -> Dict[str, bool]:
    """
    Decode PyMuPDF font flags to extract formatting properties
    """
    return {
        'superscript': bool(flags & 1),
        'italic': bool(flags & 2),
        'serif': not bool(flags & 4),
        'monospaced': bool(flags & 8),
        'bold': bool(flags & 16),  # Primary bold detection
        'has_weights': bool(flags & 32)
    }


def analyze_text_patterns(text: str) -> Dict[str, Any]:
    """
    Rule-based text analysis for heading detection (25% component)
    """
    text = text.strip()
    
    patterns = {
        'ends_with_colon': text.endswith(':'),
        'starts_with_number': bool(re.match(r'^\d+\.?\s', text)),
        'is_all_caps': text.isupper() and len(text) > 1,
        'has_roman_numeral': bool(re.match(r'^[IVX]+\.?\s', text)),
        'word_count': len(text.split()),
        'char_count': len(text),
        'has_section_keywords': any(keyword in text.lower() for keyword in 
                                   ['chapter', 'section', 'introduction', 'conclusion', 'summary']),
        'is_title_case': text.istitle()
    }
    
    return patterns


def calculate_heading_score(span_data: Dict, config: Dict) -> float:
    """
    Calculate heading likelihood score (hybrid approach)
    Combines rule-based (25%) and prepares for ML features (75%)
    FIXED: Use correct config paths
    """
    score = 0.0
    
    # Rule-based scoring (25% weight)
    font_flags = decode_font_flags(span_data['flags'])
    text_patterns = analyze_text_patterns(span_data['text'])
    
    # Font size rule (relative to document average)
    if 'avg_font_size' in span_data:
        size_ratio = span_data['size'] / span_data['avg_font_size']
        
        # FIXED: Use hierarchy config section
        hierarchy_config = config.get('hierarchy', {})
        if size_ratio >= hierarchy_config.get('size_ratio_h1', 1.5):
            score += 3.0
        elif size_ratio >= hierarchy_config.get('size_ratio_h2', 1.25):
            score += 2.0
        elif size_ratio >= hierarchy_config.get('size_ratio_h3', 1.1):
            score += 1.0
    
    # Bold text bonus
    if font_flags['bold']:
        score += 2.0
    
    # Pattern-based scoring - FIXED: Use correct config paths
    features_config = config.get('features', {})
    if text_patterns['ends_with_colon']:
        score += features_config.get('colon_bonus', 1.25)
    if text_patterns['starts_with_number']:
        score += features_config.get('number_pattern_bonus', 1.2)
    if text_patterns['is_all_caps'] and text_patterns['word_count'] <= 5:
        score += 1.5
    if text_patterns['has_section_keywords']:
        score += 1.0
    
    # Length penalties
    hierarchy_config = config.get('hierarchy', {})
    if text_patterns['word_count'] > 15:
        score -= 2.0
    if text_patterns['char_count'] > hierarchy_config.get('max_heading_length', 200):
        score -= 3.0
    
    return score


def extract_title_from_first_page(page_data: List[Dict], config: Dict) -> str:
    """
    Extract document title from first page using hybrid approach
    Find biggest, boldest text as specified in requirements
    """
    if not page_data:
        return ""
    
    # Find largest font size
    max_size = max(span['size'] for span in page_data)
    
    # Filter candidates: largest size + bold preference
    title_candidates = []
    hierarchy_config = config.get('hierarchy', {})
    
    for span in page_data:
        if span['size'] >= max_size * 0.9:  # Within 90% of max size
            font_flags = decode_font_flags(span['flags'])
            patterns = analyze_text_patterns(span['text'])
            
            # Score potential titles
            title_score = span['size']
            if font_flags['bold']:
                title_score += 5
            if patterns['word_count'] <= hierarchy_config.get('title_max_words', 15):
                title_score += 2
            if not patterns['ends_with_colon']:
                title_score += 1
                
            title_candidates.append({
                'text': span['text'].strip(),
                'score': title_score,
                'size': span['size']
            })
    
    if title_candidates:
        # Sort by score and return best candidate
        title_candidates.sort(key=lambda x: x['score'], reverse=True)
        return title_candidates[0]['text']
    
    return ""


def clean_text_for_heading(text: str) -> str:
    """Clean and normalize text for heading classification"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\w\s\-:().,]', '', text)
    
    return text


def is_potential_heading(text: str, config: Dict) -> bool:
    """
    Quick filter for potential headings using rules
    FIXED: Use correct config paths
    """
    text = text.strip()
    
    if not text:
        return False
    
    hierarchy_config = config.get('hierarchy', {})
        
    # Length checks
    if len(text) < hierarchy_config.get('min_heading_length', 3):
        return False
    if len(text) > hierarchy_config.get('max_heading_length', 200):
        return False
    
    # Word count check  
    word_count = len(text.split())
    if word_count > 25:  # Too long to be a heading
        return False
    
    # Check for sentence-ending punctuation (unlikely in headings)
    if text.endswith(('.', '!', '?')) and word_count > 8:
        return False
    
    return True


def save_json_output(title: str, outline: List[Dict], output_path: str):
    """
    Save results in Adobe Round 1A specified format
    """
    import json
    
    result = {
        "title": title,
        "outline": outline
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)