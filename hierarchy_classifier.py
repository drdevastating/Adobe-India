"""
Adobe Round 1A: Improved Hierarchy Classification Engine
FIXED: Much more selective heading detection using advanced NLP techniques
Only extracts genuine headings, not body text
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import spacy
from utils import (load_config, decode_font_flags, analyze_text_patterns, 
                   calculate_heading_score, clean_text_for_heading)

# Try to load spacy model, fallback if not available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class ImprovedHierarchyClassifier:
    """
    Improved classifier that's much more selective about headings
    Uses advanced NLP techniques and stricter filtering
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # More restrictive heading patterns
        self.heading_patterns = {
            'chapter': re.compile(r'^(chapter|chap\.?)\s+\d+', re.IGNORECASE),
            'section': re.compile(r'^(section|sec\.?)\s+\d+', re.IGNORECASE),
            'numbered': re.compile(r'^\d+(\.\d+)*\.?\s+[A-Z]', re.IGNORECASE),
            'roman': re.compile(r'^[IVX]+\.?\s+[A-Z]', re.IGNORECASE),
            'lettered': re.compile(r'^[A-Z]\.?\s+[A-Z]', re.IGNORECASE),
            'appendix': re.compile(r'^appendix\s+[A-Z\d]', re.IGNORECASE),
            'introduction': re.compile(r'^(introduction|conclusion|summary|abstract|overview)$', re.IGNORECASE),
            'bibliography': re.compile(r'^(bibliography|references|works\s+cited)$', re.IGNORECASE)
        }
        
        # Common non-heading patterns to exclude
        self.exclusion_patterns = {
            'page_numbers': re.compile(r'^\d+$'),
            'dates': re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'),
            'urls': re.compile(r'http[s]?://|www\.|\.(com|org|edu|gov)', re.IGNORECASE),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'complete_sentences': re.compile(r'^[A-Z][a-z]+.*[.!?]$'),  # Complete sentences
            'long_text': re.compile(r'^.{150,}$'),  # Very long text
            'all_lowercase': re.compile(r'^[a-z\s]+$'),  # All lowercase (likely body text)
            'mixed_case_sentence': re.compile(r'^[A-Z][a-z]+\s+[a-z]+.*[.!?]$')  # Regular sentences
        }
        
        # Label mapping - more conservative
        self.label_map = {
            'H1': 0,
            'H2': 1, 
            'H3': 2,
            'BODY': 3
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def _is_likely_heading(self, span_data: Dict) -> bool:
        """
        Pre-filter: Use strict rules to determine if text could possibly be a heading
        This dramatically reduces false positives
        """
        text = span_data.get('text', '').strip()
        
        if not text or len(text) < 2:
            return False
        
        # Immediate exclusions
        for pattern_name, pattern in self.exclusion_patterns.items():
            if pattern.search(text):
                return False
        
        # Must be relatively short
        word_count = len(text.split())
        if word_count > 12:  # Headings are typically short
            return False
        
        # Must not end with sentence punctuation (unless it's a colon)
        if text.endswith(('.', '!', '?')) and not text.endswith(':'):
            return False
        
        # Check font characteristics
        font_flags = decode_font_flags(span_data.get('flags', 0))
        size_ratio = span_data.get('size', 12.0) / span_data.get('avg_font_size', 12.0)
        
        # Must have distinctive formatting OR match heading patterns
        has_distinctive_format = (
            font_flags.get('bold', False) or 
            size_ratio >= 1.15 or  # At least 15% larger than average
            text.endswith(':')
        )
        
        has_heading_pattern = any(pattern.search(text) for pattern in self.heading_patterns.values())
        
        if not (has_distinctive_format or has_heading_pattern):
            return False
        
        # Use NLP if available for additional filtering
        if nlp and word_count > 1:
            return self._nlp_heading_check(text)
        
        return True
    
    def _nlp_heading_check(self, text: str) -> bool:
        """
        Use NLP to check if text structure suggests a heading
        """
        try:
            doc = nlp(text)
            
            # Check for complete sentences (less likely to be headings)
            sentences = list(doc.sents)
            if len(sentences) > 1:  # Multiple sentences
                return False
            
            # Check POS tags - headings usually have specific patterns
            pos_tags = [token.pos_ for token in doc if not token.is_punct]
            
            # Headings often start with nouns, proper nouns, or numbers
            if pos_tags and pos_tags[0] not in ['NOUN', 'PROPN', 'NUM', 'ADJ']:
                # Check for common heading starters
                first_word = doc[0].text.lower()
                if first_word not in ['chapter', 'section', 'introduction', 'conclusion', 'summary', 'appendix']:
                    return False
            
            # Check for verb patterns that suggest body text
            verbs = [token for token in doc if token.pos_ == 'VERB']
            if len(verbs) > 1:  # Multiple verbs suggest complex sentence
                return False
            
            return True
            
        except Exception as e:
            # Fallback to True if NLP fails
            return True
    
    def _calculate_heading_confidence(self, span_data: Dict) -> float:
        """
        Calculate confidence score for heading classification
        Higher threshold means more selective
        """
        text = span_data.get('text', '').strip()
        score = 0.0
        
        # Font-based scoring
        font_flags = decode_font_flags(span_data.get('flags', 0))
        size_ratio = span_data.get('size', 12.0) / span_data.get('avg_font_size', 12.0)
        
        # Strong indicators
        if font_flags.get('bold', False):
            score += 3.0
        if size_ratio >= 1.3:  # 30% larger
            score += 3.0
        elif size_ratio >= 1.15:  # 15% larger
            score += 1.5
        
        # Pattern-based scoring
        for pattern_name, pattern in self.heading_patterns.items():
            if pattern.search(text):
                if pattern_name in ['chapter', 'section']:
                    score += 4.0  # Very strong indicators
                elif pattern_name == 'numbered':
                    score += 2.5
                else:
                    score += 1.5
                break
        
        # Position-based scoring
        if span_data.get('page', 1) == 1:  # First page
            score += 0.5
        
        # Text characteristics
        if text.endswith(':'):
            score += 1.5
        if text.isupper() and len(text.split()) <= 4:
            score += 1.0
        if text.istitle():
            score += 0.5
        
        # Length penalty
        word_count = len(text.split())
        if word_count > 8:
            score -= 2.0
        elif word_count <= 3:
            score += 0.5
        
        return score
    
    def classify_span(self, span_data: Dict) -> Tuple[str, float]:
        """
        Classify a single span with much stricter criteria
        """
        text = span_data.get('text', '').strip()
        
        # Pre-filter: Must pass initial heading likelihood test
        if not self._is_likely_heading(span_data):
            return 'BODY', 0.0
        
        # Calculate confidence score
        confidence = self._calculate_heading_confidence(span_data)
        
        # Apply strict confidence threshold
        if confidence < 3.0:  # Minimum threshold for any heading
            return 'BODY', 0.0
        
        # Classify heading level based on font size and patterns
        size_ratio = span_data.get('size', 12.0) / span_data.get('avg_font_size', 12.0)
        font_flags = decode_font_flags(span_data.get('flags', 0))
        
        # H1: Large, bold, or clear chapter/major section indicators
        if (size_ratio >= 1.4 and font_flags.get('bold', False)) or \
           any(self.heading_patterns[p].search(text) for p in ['chapter', 'introduction', 'bibliography']):
            return 'H1', min(confidence / 5.0, 1.0)
        
        # H2: Medium size, bold, or numbered sections
        elif (size_ratio >= 1.2 and font_flags.get('bold', False)) or \
             self.heading_patterns['numbered'].search(text) or \
             self.heading_patterns['section'].search(text):
            return 'H2', min(confidence / 6.0, 1.0)
        
        # H3: Smaller headings, colon endings, or other patterns
        elif confidence >= 4.0:
            return 'H3', min(confidence / 7.0, 1.0)
        
        else:
            return 'BODY', 0.0
    
    def classify_document_spans(self, spans_data: List[Dict]) -> List[Dict]:
        """
        Classify document spans with hierarchical consistency checks
        """
        if not spans_data:
            return []
        
        # First pass: classify individual spans
        classified_spans = []
        font_size_groups = {}
        
        for span_data in spans_data:
            level, confidence = self.classify_span(span_data)
            
            if level != 'BODY':  # Only keep actual headings
                result = {
                    'text': clean_text_for_heading(span_data.get('text', '')),
                    'level': level,
                    'confidence': confidence,
                    'page': span_data.get('page', 1),
                    'font_size': span_data.get('size', 12.0),
                    'bold': span_data.get('bold', False),
                    'original_span': span_data  # Keep for consistency checks
                }
                classified_spans.append(result)
                
                # Group by font size for consistency
                font_key = f"{span_data.get('size', 12.0):.1f}_{span_data.get('bold', False)}"
                if font_key not in font_size_groups:
                    font_size_groups[font_key] = []
                font_size_groups[font_key].append(result)
        
        # Second pass: enforce hierarchical consistency
        classified_spans = self._enforce_hierarchy_rules(classified_spans, font_size_groups)
        
        # Third pass: final filtering to remove outliers
        classified_spans = self._final_filtering(classified_spans)
        
        return classified_spans
    
    def _enforce_hierarchy_rules(self, spans: List[Dict], font_groups: Dict) -> List[Dict]:
        """
        Enforce consistent hierarchy based on font sizes and patterns
        """
        if not spans:
            return spans
        
        # Sort by font size to establish hierarchy
        spans.sort(key=lambda x: x['font_size'], reverse=True)
        
        # Reassign levels based on font size hierarchy
        unique_sizes = sorted(set(span['font_size'] for span in spans), reverse=True)
        
        # Limit to top 3 different sizes for H1, H2, H3
        size_to_level = {}
        for i, size in enumerate(unique_sizes[:3]):
            if i == 0:
                size_to_level[size] = 'H1'
            elif i == 1:
                size_to_level[size] = 'H2'
            else:
                size_to_level[size] = 'H3'
        
        # Apply consistent levels
        for span in spans:
            if span['font_size'] in size_to_level:
                span['level'] = size_to_level[span['font_size']]
        
        return spans
    
    def _final_filtering(self, spans: List[Dict]) -> List[Dict]:
        """
        Final pass to remove any remaining false positives
        """
        filtered_spans = []
        
        for span in spans:
            text = span['text']
            
            # Remove very similar headings (likely duplicates)
            is_duplicate = False
            for existing in filtered_spans:
                if self._text_similarity(text, existing['text']) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Apply final quality checks
                if self._final_quality_check(span):
                    filtered_spans.append(span)
        
        return filtered_spans
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity to detect duplicates
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _final_quality_check(self, span: Dict) -> bool:
        """
        Final quality check for headings
        """
        text = span['text']
        
        # Must have reasonable length
        if len(text) < 3 or len(text) > 100:
            return False
        
        # Must not be all numbers or symbols
        if re.match(r'^[\d\s\-\.\(\)]+$', text):
            return False
        
        # Must have at least one letter
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # High confidence headings always pass
        if span['confidence'] > 0.8:
            return True
        
        # Lower confidence headings need additional validation
        word_count = len(text.split())
        if word_count > 6 and span['confidence'] < 0.6:
            return False
        
        return True


# Backward compatibility alias
HierarchyClassifier = ImprovedHierarchyClassifier


if __name__ == "__main__":
    # Test the improved classifier
    classifier = ImprovedHierarchyClassifier()
    
    # Test samples
    test_samples = [
        {
            'text': 'Chapter 1: Introduction',
            'size': 16.0,
            'flags': 16,
            'bold': True,
            'page': 1,
            'avg_font_size': 12.0
        },
        {
            'text': 'This is a long sentence that should not be classified as a heading.',
            'size': 12.0,
            'flags': 0,
            'bold': False,
            'page': 1,
            'avg_font_size': 12.0
        },
        {
            'text': '1.1 Overview',
            'size': 14.0,
            'flags': 16,
            'bold': True,
            'page': 1,
            'avg_font_size': 12.0
        }
    ]
    
    print("Testing Improved Hierarchy Classifier:")
    for i, sample in enumerate(test_samples):
        level, confidence = classifier.classify_span(sample)
        print(f"Sample {i+1}: '{sample['text'][:30]}...' -> {level} (confidence: {confidence:.3f})")