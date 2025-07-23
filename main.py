"""
Adobe Round 1A: Main Processing Pipeline
Orchestrates PDF processing for Round 1A challenge
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Import our modules
from pdf_extractor import PDFTextExtractor
from hierarchy_classifier import HierarchyClassifier
from utils import load_config, save_json_output


class Round1AProcessor:
    """
    Main processor for Adobe Round 1A challenge
    Coordinates extraction, classification, and output generation
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.extractor = PDFTextExtractor(config_path)
        self.classifier = HierarchyClassifier(config_path)
        
        # Ensure output directory exists
        self.output_dir = Path(self.config['output_folder'])
        self.output_dir.mkdir(exist_ok=True)
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF through the complete pipeline
        Returns processing result with timing information
        """
        start_time = time.time()
        pdf_name = Path(pdf_path).stem
        
        try:
            print(f"\n🔍 Processing: {os.path.basename(pdf_path)}")
            
            # Step 1: Extract text and font metadata
            print("  📝 Extracting text and font metadata...")
            text_data, title = self.extractor.extract_text_with_metadata(pdf_path)
            
            if not text_data:
                return {
                    'file': os.path.basename(pdf_path),
                    'status': 'no_text_found',
                    'processing_time': time.time() - start_time
                }
            
            print(f"     ✓ Extracted {len(text_data)} text elements")
            print(f"     ✓ Detected title: '{title}'")
            
            # Step 2: Classify text hierarchy using ML + Rules
            print("  🧠 Classifying text hierarchy (75% ML + 25% Rules)...")
            classified_spans = self.classifier.classify_document_spans(text_data)
            
            # Step 3: Filter and format outline according to Round 1A spec
            outline = self._generate_outline(classified_spans)
            
            print(f"     ✓ Identified {len(outline)} heading elements")
            
            # Step 4: Generate Round 1A JSON output
            output_file = self.output_dir / f"{pdf_name}.json"
            save_json_output(title, outline, str(output_file))
            
            processing_time = time.time() - start_time
            
            print(f"  ✅ Completed in {processing_time:.2f}s")
            print(f"     📄 Output saved: {output_file}")
            
            return {
                'file': os.path.basename(pdf_path),
                'status': 'success',
                'title': title,
                'outline_elements': len(outline),
                'total_text_elements': len(text_data),
                'processing_time': processing_time,
                'output_file': str(output_file)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"  ❌ Error: {str(e)}")
            
            return {
                'file': os.path.basename(pdf_path),
                'status': 'error',
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _generate_outline(self, classified_spans: List[Dict]) -> List[Dict]:
        """
        Generate outline in Round 1A format from classified spans
        Format: [{"level": "H1", "text": "heading text", "page": 1}, ...]
        """
        outline = []
        
        for span in classified_spans:
            # Only include headings (not BODY or TITLE)
            if span['level'] in ['H1', 'H2', 'H3']:
                outline_item = {
                    "level": span['level'],
                    "text": span['text'],
                    "page": span['page']
                }
                outline.append(outline_item)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_outline = []
        for item in outline:
            key = (item['level'], item['text'], item['page'])
            if key not in seen:
                seen.add(key)
                unique_outline.append(item)
        
        return unique_outline
    
    def process_all_pdfs(self) -> Dict:
        """
        Process all PDFs in input directory
        Main entry point for Round 1A processing
        """
        input_dir = Path(self.config['input_folder'])
        
        if not input_dir.exists():
            print(f"❌ Input directory '{input_dir}' does not exist!")
            return {'status': 'error', 'message': 'Input directory not found'}
        
        pdf_files = list(input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in '{input_dir}'")
            return {'status': 'error', 'message': 'No PDF files found'}
        
        print(f"🚀 Adobe Round 1A Processing Started")
        print(f"📁 Input directory: {input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Found {len(pdf_files)} PDF file(s)")
        
        # Process each PDF
        results = []
        successful = 0
        failed = 0
        total_time = time.time()
        
        for pdf_path in pdf_files:
            result = self.process_single_pdf(str(pdf_path))
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
            
            # Check timeout constraint (10 seconds per PDF)
            if result['processing_time'] > self.config['processing_timeout']:
                print(f"⚠️  Warning: {result['file']} exceeded {self.config['processing_timeout']}s timeout")
        
        total_processing_time = time.time() - total_time
        
        # Generate summary
        summary = {
            'status': 'completed',
            'total_pdfs': len(pdf_files),
            'successful': successful,
            'failed': failed,
            'total_time': total_processing_time,
            'average_time_per_pdf': total_processing_time / len(pdf_files),
            'results': results
        }
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = self.output_dir / "processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"🏁 ADOBE ROUND 1A PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {successful} files")
        print(f"❌ Failed: {failed} files")
        print(f"⏱️  Total time: {total_processing_time:.2f}s")
        print(f"⏱️  Average per PDF: {total_processing_time/len(pdf_files):.2f}s")
        print(f"📋 Summary saved: {summary_path}")
        
        if successful > 0:
            print(f"\n📄 Generated JSON files:")
            for result in results:
                if result['status'] == 'success':
                    print(f"  • {result['output_file']}")
        
        return summary


def main():
    """
    Main entry point for Docker container execution
    """
    print("🎯 Adobe Round 1A: PDF Hierarchy Detection")
    print("🔧 Hybrid ML (75%) + Rule-based (25%) Approach")
    print("-" * 50)
    
    # Initialize processor
    processor = Round1AProcessor()
    
    # Process all PDFs
    results = processor.process_all_pdfs()
    
    # Exit with appropriate code
    if results['status'] == 'completed' and results['failed'] == 0:
        print("\n🎉 All files processed successfully!")
        sys.exit(0)
    elif results['status'] == 'completed':
        print(f"\n⚠️  Processing completed with {results['failed']} failures")
        sys.exit(1)
    else:
        print(f"\n💥 Processing failed: {results.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()