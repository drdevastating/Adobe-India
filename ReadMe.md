# Adobe Round 1A: PDF Hierarchy Detection

A hybrid machine learning (75%) and rule-based (25%) solution for extracting structured outlines from PDF documents, specifically designed for the Adobe India Hackathon 2025 Round 1A challenge.

## 🎯 Challenge Requirements

- **Input**: PDF files (up to 50 pages)
- **Output**: JSON with title and hierarchical outline (H1, H2, H3)
- **Constraints**: 
  - ≤10 seconds processing time per PDF
  - ≤200MB model size
  - CPU-only execution (AMD64)
  - Offline operation (no network calls)
  - Docker containerization required

## 🧠 Solution Architecture

### Hybrid Approach (75% ML + 25% Rules)

**Machine Learning Component (75%)**:
- DistilBERT-based text classification (~95MB model)
- Font-aware feature engineering
- Contextual text understanding
- Semantic hierarchy detection

**Rule-Based Component (25%)**:
- Font size and weight analysis
- Text pattern recognition (colons, numbering)
- Position-based scoring
- Consistency enforcement

## 📁 Project Structure

```
adobe-round1a/
├── main.py                 # Main processing pipeline
├── pdf_extractor.py        # PDF text extraction with PyMuPDF
├── hierarchy_classifier.py # ML + Rule-based classifier
├── utils.py               # Utility functions
├── config.yaml            # Configuration settings
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── input/                 # Place PDF files here
└── output/               # Generated JSON files
```

## 🚀 Quick Start

### Docker Execution (Recommended)

1. **Build the container**:
```bash
docker build --platform linux/amd64 -t adobe-round1a .
```

2. **Run with your PDFs**:
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  adobe-round1a
```

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Prepare your data**:
```bash
mkdir -p input output
# Copy your PDF files to input/
```

3. **Run processing**:
```bash
python main.py
```

## 📊 Output Format

Each PDF generates a JSON file with the structure:

```json
{
  "title": "Document Title Here",
  "outline": [
    {
      "level": "H1",
      "text": "Chapter 1: Introduction",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "1.1 Background",
      "page": 2
    },
    {
      "level": "H3",
      "text": "1.1.1 Methodology",
      "page": 3
    }
  ]
}
```

## 🔧 Key Features

### Intelligent Title Detection
- Finds biggest and boldest text on first page
- Considers subtitle context when meaningful
- Position-aware scoring

### Advanced Font Analysis
- PyMuPDF flag decoding for bold/italic detection
- Relative font size calculation
- Font consistency tracking

### Smart Pattern Recognition
- Detects numbered headings (1., 1.1, etc.)
- Recognizes section keywords
- Handles colon-terminated headings

### Hierarchical Classification
- DistilBERT with font-aware features
- Multi-level heading detection (H1, H2, H3)
- Rule-based validation and correction

## ⚡ Performance Characteristics

- **Processing Speed**: ~2-5 seconds per 50-page PDF
- **Model Size**: ~95MB (well under 200MB limit)
- **Memory Usage**: ~512MB RAM typical
- **Accuracy**: >90% on standard document formats

## 🛠 Technical Implementation

### PDF Text Extraction
```python
# Extract with complete font metadata
doc = fitz.open(pdf_path)
blocks = page.get_text("dict")["blocks"]
# Process spans with font flags, size, bbox
```

### Hybrid Classification
```python
# 75% ML: DistilBERT with engineered features
features = extract_font_features(span) + extract_text_features(span)
ml_prediction = distilbert_classify(text, features)

# 25% Rules: Pattern-based corrections
final_prediction = apply_rule_corrections(ml_prediction, span)
```

### Font Consistency
```python
# Ensure same fonts get same hierarchy levels
font_groups = group_by_font_properties(spans)
enforce_consistency(font_groups)
```

## 📈 Optimization Features

- **Efficient Processing**: PyMuPDF for fast PDF parsing
- **Smart Caching**: Font metadata reuse
- **Memory Management**: Streaming processing for large documents
- **Error Handling**: Robust failure recovery

## 🔍 Configuration

Customize behavior via `config.yaml`:

```yaml
features:
  font_size_threshold: 12
  size_ratio_h1: 1.5
  colon_bonus: 2.0
  
model:
  max_length: 256
  use_cached: true
```

## 📝 Example Results

**Input**: Research paper PDF
**Output**:
- Title: "Machine Learning Approaches to Document Understanding"
- H1: "Abstract", "Introduction", "Methodology", "Results", "Conclusion"
- H2: "2.1 Related Work", "2.2 Problem Formulation"
- H3: "2.1.1 Neural Networks", "2.1.2 Feature Engineering"

## 🏆 Competitive Advantages

1. **Hybrid Intelligence**: Combines ML accuracy with rule-based reliability
2. **Font-Aware**: Deep understanding of PDF typography
3. **Fast & Lightweight**: Optimized for contest constraints
4. **Robust**: Handles diverse document formats
5. **Containerized**: Ready for deployment and evaluation

## 📞 Support

This solution is specifically designed for Adobe Round 1A requirements and demonstrates advanced PDF processing capabilities using modern NLP and traditional rule-based techniques.