"""
Adobe Round 1A: Hierarchy Classification Engine
75% ML + 25% Rule-based approach for PDF heading detection
Uses DistilBERT for text classification with font features
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils import (load_config, decode_font_flags, analyze_text_patterns, 
                   calculate_heading_score, clean_text_for_heading)


class HierarchyClassifier:
    """
    Hybrid ML + Rule-based classifier for PDF heading hierarchy
    75% ML (DistilBERT) + 25% Rules for heading level detection
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Label mapping for hierarchy levels
        self.label_map = {
            'TITLE': 0,
            'H1': 1, 
            'H2': 2,
            'H3': 3,
            'BODY': 4
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def _initialize_model(self):
        """Initialize DistilBERT model for text classification"""
        model_name = self.config['model']['name']
        
        # Load or create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load or create model
        try:
            # Try to load pre-trained model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "models/hierarchy_classifier",
                num_labels=len(self.label_map)
            )
            print("✓ Loaded pre-trained hierarchy model")
        except:
            # Create new model if no pre-trained exists
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.label_map)
            )
            print("✓ Initialized new DistilBERT model")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _extract_features(self, span_data: Dict) -> np.ndarray:
        """
        Extract engineered features for ML model (75% component)
        Combines text features with font/layout features
        """
        text = span_data['text']
        
        # Text pattern features (rule-based part)
        patterns = analyze_text_patterns(text)
        
        # Font features
        font_flags = decode_font_flags(span_data['flags'])
        
        # Positional features
        bbox = span_data['bbox']
        
        features = [
            # Font features
            span_data['size'],
            span_data.get('avg_font_size', 12),
            span_data['size'] / span_data.get('avg_font_size', 12),  # Relative size
            float(font_flags['bold']),
            float(font_flags['italic']),
            
            # Text pattern features
            float(patterns['ends_with_colon']),
            float(patterns['starts_with_number']),
            float(patterns['is_all_caps']),
            float(patterns['has_roman_numeral']),
            patterns['word_count'],
            patterns['char_count'],
            float(patterns['has_section_keywords']),
            float(patterns['is_title_case']),
            
            # Position features
            bbox[0],  # x0
            bbox[1],  # y0
            bbox[2] - bbox[0],  # width
            bbox[3] - bbox[1],  # height
            
            # Page features
            span_data['page'],
            
            # Rule-based heading score (25% weight)
            calculate_heading_score(span_data, self.config)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _prepare_text_for_bert(self, span_data: Dict) -> str:
        """
        Prepare text input for DistilBERT with font context
        """
        text = clean_text_for_heading(span_data['text'])
        
        # Add font information as special tokens
        font_flags = decode_font_flags(span_data['flags'])
        font_info = []
        
        if font_flags['bold']:
            font_info.append("[BOLD]")
        if font_flags['italic']:
            font_info.append("[ITALIC]")
        
        size_info = f"[SIZE:{span_data['size']:.1f}]"
        
        # Combine text with font context
        context_text = f"{' '.join(font_info)} {size_info} {text}"
        
        return context_text
    
    def classify_span(self, span_data: Dict) -> Tuple[str, float]:
        """
        Classify a single text span using hybrid approach
        Returns (predicted_level, confidence_score)
        """
        if not self.model:
            self._initialize_model()
        
        text = span_data['text'].strip()
        
        # Quick rule-based filtering
        if not span_data.get('is_potential_heading', True):
            return 'BODY', 0.0
        
        # Extract features for ML
        features = self._extract_features(span_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Prepare text for BERT
        bert_text = self._prepare_text_for_bert(span_data)
        
        # Tokenize text
        inputs = self.tokenizer(
            bert_text,
            truncation=True,
            padding=True,
            max_length=self.config['model']['max_length'],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get ML prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_level = self.reverse_label_map[predicted_class]
        
        # Apply rule-based corrections (25% component)
        predicted_level = self._apply_rule_corrections(span_data, predicted_level, confidence)
        
        return predicted_level, confidence
    
    def _apply_rule_corrections(self, span_data: Dict, ml_prediction: str, confidence: float) -> str:
        """
        Apply rule-based corrections to ML predictions (25% component)
        """
        text = span_data['text']
        patterns = analyze_text_patterns(text)
        font_flags = decode_font_flags(span_data['flags'])
        
        # Rule 1: Text ending with ":" is likely a heading
        if patterns['ends_with_colon'] and ml_prediction == 'BODY':
            if patterns['word_count'] <= 8:
                return 'H3'
        
        # Rule 2: Very large font + bold is likely H1 or Title
        size_ratio = span_data['size'] / span_data.get('avg_font_size', 12)
        if size_ratio > 1.8 and font_flags['bold']:
            if span_data['page'] == 1 and ml_prediction not in ['TITLE', 'H1']:
                return 'H1'
        
        # Rule 3: Numbered patterns suggest headings
        if patterns['starts_with_number'] and ml_prediction == 'BODY':
            if patterns['word_count'] <= 12:
                return 'H2'
        
        # Rule 4: Section keywords boost heading likelihood
        if patterns['has_section_keywords'] and ml_prediction == 'BODY':
            if font_flags['bold'] or size_ratio > 1.2:
                return 'H3'
        
        # Rule 5: Very long text is unlikely to be a heading
        if patterns['word_count'] > 20:
            return 'BODY'
        
        return ml_prediction
    
    def classify_document_spans(self, spans_data: List[Dict]) -> List[Dict]:
        """
        Classify all spans in a document and determine hierarchy
        """
        if not spans_data:
            return []
        
        classified_spans = []
        font_groups = {}  # Track consistent font patterns
        
        for span_data in spans_data:
            level, confidence = self.classify_span(span_data)
            
            result = {
                'text': clean_text_for_heading(span_data['text']),
                'level': level,
                'confidence': confidence,
                'page': span_data['page'],
                'font_size': span_data['size'],
                'bold': span_data['bold']
            }
            
            # Track font consistency (25% rule component)
            font_key = f"{span_data['font']}_{span_data['size']:.1f}_{span_data['bold']}"
            if font_key not in font_groups:
                font_groups[font_key] = []
            font_groups[font_key].append(result)
            
            classified_spans.append(result)
        
        # Apply font consistency rules
        classified_spans = self._enforce_font_consistency(classified_spans, font_groups)
        
        return classified_spans
    
    def _enforce_font_consistency(self, spans: List[Dict], font_groups: Dict) -> List[Dict]:
        """
        Ensure consistent font properties get same hierarchy levels (Rule component)
        """
        # For each font group, determine the most common level
        for font_key, group_spans in font_groups.items():
            if len(group_spans) > 1:
                # Find most common level in this font group
                levels = [span['level'] for span in group_spans if span['level'] != 'BODY']
                if levels:
                    from collections import Counter
                    most_common_level = Counter(levels).most_common(1)[0][0]
                    
                    # Update all spans in this group (if they're headings)
                    for span in group_spans:
                        if span['level'] != 'BODY' and span['confidence'] < 0.8:
                            span['level'] = most_common_level
        
        return spans


if __name__ == "__main__":
    # Test the classifier
    classifier = HierarchyClassifier()
    
    # Example usage
    sample_span = {
        'text': 'Chapter 1: Introduction',
        'size': 16.0,
        'flags': 16,  # Bold flag
        'bold': True,
        'italic': False,
        'bbox': [50, 100, 200, 120],
        'page': 1,
        'avg_font_size': 12.0,
        'is_potential_heading': True
    }
    
    level, confidence = classifier.classify_span(sample_span)
    print(f"Predicted level: {level}, Confidence: {confidence:.3f}")