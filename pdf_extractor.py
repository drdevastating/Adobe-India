"""
Adobe Round 1A: PDF Text Extraction Engine
Extracts all text with font metadata using PyMuPDF
"""

import fitz  # PyMuPDF
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
from utils import load_config, decode_font_flags, is_potential_heading, clean_text_for_heading


class PDFTextExtractor:
    """
    Extracts text and font information from PDF files
    Implements the core extraction logic for Round 1A
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.input_dir = Path(self.config['input_folder'])
        self.output_dir = Path(self.config['output_folder'])
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_text_with_metadata(self, pdf_path: str) -> Tuple[List[Dict], str]:
        """
        Extract all text spans with complete font and position metadata
        Returns (text_data, title)
        """
        doc = fitz.open(pdf_path)
        all_spans = []
        first_page_spans = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block_idx, block in enumerate(blocks):
                    if "lines" not in block:
                        continue
                        
                    for line_idx, line in enumerate(block["lines"]):
                        for span_idx, span in enumerate(line["spans"]):
                            text = span["text"].strip()
                            if not text:
                                continue
                            
                            # Decode font flags
                            font_flags = decode_font_flags(span["flags"])
                            
                            span_data = {
                                'page': page_num + 1,
                                'block_idx': block_idx,
                                'line_idx': line_idx,
                                'span_idx': span_idx,
                                'text': text,
                                'font': span["font"],
                                'size': round(span["size"], 2),
                                'flags': span["flags"],
                                'bold': font_flags['bold'],
                                'italic': font_flags['italic'],
                                'bbox': span["bbox"],
                                'color': span.get("color", 0),
                                'is_potential_heading': is_potential_heading(text, self.config)
                            }
                            
                            all_spans.append(span_data)
                            
                            # Collect first page spans for title detection
                            if page_num == 0:
                                first_page_spans.append(span_data)
        
        finally:
            doc.close()
        
        # Calculate average font size for relative sizing
        if all_spans:
            avg_size = sum(span['size'] for span in all_spans) / len(all_spans)
            for span in all_spans:
                span['avg_font_size'] = avg_size
            for span_fp in first_page_spans:
                span_fp['avg_font_size'] = avg_size
        
        # Extract title from first page
        title = self._extract_title(first_page_spans)
        
        return all_spans, title
    
    def _extract_title(self, first_page_spans: List[Dict]) -> str:
        """
        Extract title from first page - biggest and boldest text
        """
        if not first_page_spans:
            return ""
        
        # Find the largest font size
        max_size = max(span['size'] for span in first_page_spans)
        
        # Get candidates with largest size
        large_text_candidates = [
            span for span in first_page_spans 
            if span['size'] >= max_size * 0.95  # Within 95% of max size
        ]
        
        if not large_text_candidates:
            return ""
        
        # Prefer bold text among large candidates
        bold_candidates = [span for span in large_text_candidates if span['bold']]
        
        if bold_candidates:
            candidates = bold_candidates
        else:
            candidates = large_text_candidates
        
        # Score candidates based on position and characteristics
        scored_candidates = []
        for span in candidates:
            score = span['size']  # Base score from size
            
            # Bonus for bold
            if span['bold']:
                score += 5
            
            # Bonus for position (higher on page is better)
            y_pos = span['bbox'][1]  # y0 coordinate
            if y_pos < 200:  # Top portion of page
                score += 3
            
            # Penalty for very long text (unlikely to be title)
            word_count = len(span['text'].split())
            if word_count > 15:
                score -= 5
            elif word_count <= 8:
                score += 2
            
            # Penalty for text ending with colon (likely a section header)
            if span['text'].endswith(':'):
                score -= 2
            
            scored_candidates.append((score, span['text']))
        
        # Return highest scoring candidate
        if scored_candidates:
            scored_candidates.sort(reverse=True)
            return clean_text_for_heading(scored_candidates[0][1])
        
        return ""
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF file and return extraction results
        """
        try:
            print(f"Processing: {os.path.basename(pdf_path)}")
            
            # Extract text and metadata
            text_data, title = self.extract_text_with_metadata(pdf_path)
            
            if not text_data:
                print(f"Warning: No text extracted from {pdf_path}")
                return {
                    'file': os.path.basename(pdf_path),
                    'status': 'no_text',
                    'text_elements': 0,
                    'title': "",
                    'processing_time': 0
                }
            
            # Save intermediate data for ML processing
            df = pd.DataFrame(text_data)
            intermediate_path = self.output_dir / f"{Path(pdf_path).stem}_extracted.csv"
            df.to_csv(intermediate_path, index=False)
            
            result = {
                'file': os.path.basename(pdf_path),
                'status': 'success',
                'text_elements': len(text_data),
                'title': title,
                'intermediate_file': str(intermediate_path),
                'processing_time': datetime.now().isoformat()
            }
            
            print(f"✓ Extracted {len(text_data)} text elements")
            print(f"✓ Title: {title}")
            
            return result
            
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {str(e)}")
            return {
                'file': os.path.basename(pdf_path),
                'status': 'error',
                'error': str(e),
                'processing_time': datetime.now().isoformat()
            }
    
    def process_all_pdfs(self) -> List[Dict]:
        """
        Process all PDF files in the input directory
        """
        if not self.input_dir.exists():
            print(f"Input directory {self.input_dir} does not exist!")
            return []
        
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.input_dir}")
            return []
        
        print(f"Found {len(pdf_files)} PDF file(s) to process")
        
        results = []
        for pdf_path in pdf_files:
            result = self.process_single_pdf(str(pdf_path))
            results.append(result)
        
        # Save processing summary
        summary_df = pd.DataFrame(results)
        summary_path = self.output_dir / "extraction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n📊 Processing complete! Summary saved to: {summary_path}")
        
        return results


if __name__ == "__main__":
    extractor = PDFTextExtractor()
    results = extractor.process_all_pdfs()
    
    print("\n" + "="*60)
    print("PDF TEXT EXTRACTION COMPLETE")
    print("="*60)
    
    # Categorize results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    no_text = [r for r in results if r['status'] == 'no_text']
    
    # Print summary statistics
    print(f"\n📊 Processing Summary:")
    print(f"   Total files processed: {len(results)}")
    print(f"   ✅ Successful: {len(successful)} files")
    print(f"   ❌ Failed: {len(failed)} files") 
    print(f"   ⚠️  No text found: {len(no_text)} files")
    
    if results:
        success_rate = len(successful) / len(results) * 100
        print(f"   📈 Success rate: {success_rate:.1f}%")
    
    # Report failed files with details
    if failed:
        print(f"\n❌ Failed Files:")
        for result in failed:
            print(f"   • {result['file']}: {result.get('error', 'Unknown error')}")
    
    # Report files with no text
    if no_text:
        print(f"\n⚠️  Files with no extractable text:")
        for result in no_text:
            print(f"   • {result['file']}: Empty or image-only PDF")
    
    # List successfully processed files
    if successful:
        print(f"\n✅ Successfully processed files:")
        total_elements = 0
        for result in successful:
            elements = result['text_elements']
            total_elements += elements
            title_preview = result['title'][:50] + "..." if len(result['title']) > 50 else result['title']
            print(f"   • {result['file']}: {elements:,} text elements")
            print(f"     Title: '{title_preview}'")
        
        print(f"\n📈 Total text elements extracted: {total_elements:,}")
        avg_elements = total_elements / len(successful)
        print(f"📈 Average elements per file: {avg_elements:.0f}")
    
    # Next steps guidance
    print(f"\n🔗 Next Steps:")
    if successful:
        print(f"   1. Review extracted data in: output/")
        print(f"   2. Run hierarchy classification: python hierarchy_classifier.py")
        print(f"   3. Generate final JSON: python main.py")
    else:
        print(f"   ⚠️  No files were successfully processed.")
        print(f"   Check input directory and PDF file formats.")
    
    # Exit with appropriate code
    if failed and not successful:
        print(f"\n💥 All processing failed!")
        exit(1)
    elif failed:
        print(f"\n⚠️  Processing completed with some failures.")
        exit(2)
    else:
        print(f"\n🎉 All files processed successfully!")
        exit(0)