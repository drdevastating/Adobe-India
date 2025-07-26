"""
Adobe Round 1A: Improved Main Processing Pipeline
FIXED: Now produces much cleaner, smaller JSON files with only genuine headings
Uses advanced NLP and stricter filtering
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import traceback
import json

# Import improved modules
try:
    from pdf_extractor import ImprovedPDFTextExtractor as PDFTextExtractor
    from hierarchy_classifier import ImprovedHierarchyClassifier as HierarchyClassifier
    from utils import load_config, save_json_output
except ImportError:
    try:
        # Fallback to original modules if improved ones not available
        from pdf_extractor import PDFTextExtractor
        from hierarchy_classifier import HierarchyClassifier
        from utils import load_config, save_json_output
        print("Warning: Using original modules. For best results, use improved modules.")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)


class ImprovedRound1AProcessor:
    """
    Improved processor that generates much cleaner, smaller JSON outputs
    Focus on quality over quantity
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        try:
            self.config = load_config(config_path)
            self.extractor = PDFTextExtractor(config_path)
            self.classifier = HierarchyClassifier(config_path)
            
            # Ensure output directory exists
            self.output_dir = Path(self.config.get('output_folder', 'output'))
            self.output_dir.mkdir(exist_ok=True)
            
            # Quality control parameters
            self.max_headings_per_document = self.config.get('max_headings_per_document', 20)
            self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.4)
            
        except Exception as e:
            print(f"Error initializing processor: {e}")
            # Use defaults
            self.output_dir = Path('output')
            self.output_dir.mkdir(exist_ok=True)
            self.config = {
                'processing_timeout': 8,
                'max_headings_per_document': 20,
                'min_confidence_threshold': 0.4
            }
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF with focus on quality heading extraction
        """
        start_time = time.time()
        pdf_name = Path(pdf_path).stem
        
        try:
            print(f"\n🔍 Processing: {os.path.basename(pdf_path)}")
            
            # Step 1: Extract potential headings (much more selective now)
            print("  📝 Extracting potential headings...")
            try:
                potential_headings, title = self.extractor.extract_text_with_metadata(pdf_path)
            except Exception as e:
                print(f"    ❌ Text extraction failed: {e}")
                return {
                    'file': os.path.basename(pdf_path),
                    'status': 'extraction_failed',
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
            
            if not potential_headings:
                print("    ⚠️  No potential headings found")
                # Create minimal output with just title
                self._save_minimal_output(pdf_name, title)
                return {
                    'file': os.path.basename(pdf_path),
                    'status': 'no_headings_found',
                    'title': title,
                    'outline_elements': 0,
                    'processing_time': time.time() - start_time
                }
            
            print(f"     ✓ Found {len(potential_headings)} potential heading candidates")
            print(f"     ✓ Document title: '{title}'")
            
            # Step 2: Classify headings using improved ML + Rules
            print("  🧠 Classifying headings with advanced NLP...")
            try:
                classified_spans = self.classifier.classify_document_spans(potential_headings)
            except Exception as e:
                print(f"    ❌ Classification failed: {e}")
                print(f"    🔄 Using fallback classification...")
                classified_spans = self._fallback_classification(potential_headings)
            
            # Step 3: Apply quality control filters
            print("  🔍 Applying quality control filters...")
            high_quality_headings = self._apply_quality_control(classified_spans)
            
            # Step 4: Generate final outline
            outline = self._generate_clean_outline(high_quality_headings)
            
            print(f"     ✓ Final headings: {len(outline)} high-quality headings")
            
            # Step 5: Save output
            try:
                output_file = self.output_dir / f"{pdf_name}.json"
                save_json_output(title, outline, str(output_file))
                
                # Also save a human-readable summary
                self._save_readable_summary(pdf_name, title, outline, high_quality_headings)
                
            except Exception as e:
                print(f"    ❌ Failed to save output: {e}")
                return {
                    'file': os.path.basename(pdf_path),
                    'status': 'save_failed',
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
            
            processing_time = time.time() - start_time
            
            print(f"  ✅ Completed in {processing_time:.2f}s")
            print(f"     📄 Output saved: {output_file}")
            
            return {
                'file': os.path.basename(pdf_path),
                'status': 'success',
                'title': title,
                'outline_elements': len(outline),
                'potential_headings': len(potential_headings),
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
    
    def _apply_quality_control(self, classified_spans: List[Dict]) -> List[Dict]:
        """
        Apply strict quality control to ensure only genuine headings
        """
        if not classified_spans:
            return []
        
        # Filter by confidence threshold
        high_confidence = [
            span for span in classified_spans 
            if span.get('confidence', 0) >= self.min_confidence_threshold
        ]
        
        # Remove duplicates and very similar headings
        unique_headings = []
        for span in high_confidence:
            is_duplicate = False
            for existing in unique_headings:
                similarity = self._calculate_text_similarity(span['text'], existing['text'])
                if similarity > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if span['confidence'] > existing['confidence']:
                        unique_headings.remove(existing)
                        unique_headings.append(span)
                    break
            
            if not is_duplicate:
                unique_headings.append(span)
        
        # Limit total number of headings per document
        if len(unique_headings) > self.max_headings_per_document:
            # Sort by confidence and keep top N
            unique_headings.sort(key=lambda x: x['confidence'], reverse=True)
            unique_headings = unique_headings[:self.max_headings_per_document]
            print(f"    📊 Limited to top {self.max_headings_per_document} headings")
        
        # Sort by page number and position for final output
        unique_headings.sort(key=lambda x: (x.get('page', 1), x.get('confidence', 0)), reverse=False)
        
        return unique_headings
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _fallback_classification(self, text_data: List[Dict]) -> List[Dict]:
        """
        Fallback classification when ML fails - very conservative
        """
        classified_spans = []
        
        try:
            for span_data in text_data:
                text = span_data.get('text', '').strip()
                if not text:
                    continue
                
                # Very conservative classification
                font_size = span_data.get('size', 12.0)
                avg_size = span_data.get('avg_font_size', 12.0)
                is_bold = span_data.get('bold', False)
                size_ratio = font_size / avg_size if avg_size > 0 else 1.0
                
                # Only classify as heading if very clear indicators
                confidence = 0.0
                level = 'BODY'
                
                if size_ratio >= 1.4 and is_bold:
                    level = 'H1'
                    confidence = 0.8
                elif size_ratio >= 1.25 and is_bold:
                    level = 'H2'
                    confidence = 0.7
                elif (size_ratio >= 1.15 and is_bold) or text.endswith(':'):
                    level = 'H3'
                    confidence = 0.6
                
                # Additional pattern checking
                import re
                if re.match(r'^(chapter|section)\s+\d+', text, re.IGNORECASE):
                    level = 'H1'
                    confidence = 0.9
                elif re.match(r'^\d+\.\d+', text):
                    level = 'H2'
                    confidence = 0.7
                
                if level != 'BODY' and confidence >= 0.5:
                    classified_spans.append({
                        'text': text,
                        'level': level,
                        'confidence': confidence,
                        'page': span_data.get('page', 1),
                        'font_size': font_size,
                        'bold': is_bold
                    })
            
            return classified_spans
            
        except Exception as e:
            print(f"Warning: Fallback classification failed: {e}")
            return []
    
    def _generate_clean_outline(self, headings: List[Dict]) -> List[Dict]:
        """
        Generate clean outline with proper hierarchy
        """
        outline = []
        
        for heading in headings:
            level = heading.get('level', 'H3')
            text = heading.get('text', '').strip()
            page = heading.get('page', 1)
            
            if level in ['H1', 'H2', 'H3'] and text:
                outline_item = {
                    "level": level,
                    "text": text,
                    "page": page
                }
                outline.append(outline_item)
        
        return outline
    
    def _save_minimal_output(self, pdf_name: str, title: str):
        """
        Save minimal output when no headings found
        """
        try:
            output_file = self.output_dir / f"{pdf_name}.json"
            result = {
                "title": title,
                "outline": []
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Could not save minimal output: {e}")
    
    def _save_readable_summary(self, pdf_name: str, title: str, outline: List[Dict], 
                              all_headings: List[Dict]):
        """
        Save human-readable summary for review
        """
        try:
            summary_file = self.output_dir / f"{pdf_name}_summary.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"PDF Processing Summary: {pdf_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Title: {title}\n\n")
                f.write(f"Total Headings Found: {len(outline)}\n\n")
                
                if outline:
                    f.write("Document Outline:\n")
                    f.write("-" * 20 + "\n")
                    for i, item in enumerate(outline, 1):
                        indent = "  " * (int(item['level'][1]) - 1)
                        f.write(f"{indent}{i}. {item['level']}: {item['text']} (Page {item['page']})\n")
                
                if all_headings:
                    f.write(f"\n\nDetailed Analysis:\n")
                    f.write("-" * 20 + "\n")
                    for heading in all_headings:
                        f.write(f"{heading['level']}: {heading['text']}\n")
                        f.write(f"  Confidence: {heading['confidence']:.2f}\n")
                        f.write(f"  Page: {heading['page']}, Font Size: {heading.get('font_size', 'N/A')}\n")
                        f.write(f"  Bold: {heading.get('bold', 'N/A')}\n\n")
                
        except Exception as e:
            print(f"Warning: Could not save summary: {e}")
    
    def process_all_pdfs(self) -> Dict:
        """
        Process all PDFs with improved quality control
        """
        try:
            input_dir = Path(self.config.get('input_folder', 'input'))
            
            if not input_dir.exists():
                print(f"❌ Input directory '{input_dir}' does not exist!")
                return {'status': 'error', 'message': 'Input directory not found'}
            
            pdf_files = list(input_dir.glob("*.pdf"))
            
            if not pdf_files:
                print(f"❌ No PDF files found in '{input_dir}'")
                return {'status': 'error', 'message': 'No PDF files found'}
            
            print(f"🚀 Adobe Round 1A - Improved Processing Started")
            print(f"📁 Input directory: {input_dir}")
            print(f"📁 Output directory: {self.output_dir}")
            print(f"📊 Found {len(pdf_files)} PDF file(s)")
            print(f"🎯 Quality settings: Max {self.max_headings_per_document} headings, Min confidence {self.min_confidence_threshold}")
            
            # Process each PDF
            results = []
            successful = 0
            failed = 0
            total_headings = 0
            total_time = time.time()
            
            for pdf_path in pdf_files:
                try:
                    result = self.process_single_pdf(str(pdf_path))
                    results.append(result)
                    
                    if result['status'] == 'success':
                        successful += 1
                        total_headings += result.get('outline_elements', 0)
                    else:
                        failed += 1
                        
                except Exception as e:
                    print(f"❌ Critical error processing {pdf_path}: {e}")
                    results.append({
                        'file': os.path.basename(str(pdf_path)),
                        'status': 'critical_error',
                        'error': str(e),
                        'processing_time': 0
                    })
                    failed += 1
            
            total_processing_time = time.time() - total_time
            
            # Generate summary
            summary = {
                'status': 'completed',
                'total_pdfs': len(pdf_files),
                'successful': successful,
                'failed': failed,
                'total_headings_extracted': total_headings,
                'avg_headings_per_pdf': total_headings / successful if successful > 0 else 0,
                'total_time': total_processing_time,
                'average_time_per_pdf': total_processing_time / len(pdf_files),
                'results': results
            }
            
            # Save summary
            try:
                summary_df = pd.DataFrame(results)
                summary_path = self.output_dir / "processing_summary.csv"
                summary_df.to_csv(summary_path, index=False)
            except Exception as e:
                print(f"Warning: Could not save summary CSV: {e}")
            
            # Print final summary
            print(f"\n{'='*60}")
            print(f"🏁 IMPROVED PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"✅ Successfully processed: {successful} files")
            print(f"❌ Failed: {failed} files")
            print(f"📊 Total headings extracted: {total_headings}")
            print(f"📊 Average headings per file: {total_headings/successful:.1f}" if successful > 0 else "📊 No successful files")
            print(f"⏱️  Total time: {total_processing_time:.2f}s")
            print(f"⏱️  Average per PDF: {total_processing_time/len(pdf_files):.2f}s")
            
            if successful > 0:
                print(f"\n📄 Generated JSON files:")
                for result in results:
                    if result['status'] == 'success':
                        headings_count = result.get('outline_elements', 0)
                        print(f"  • {result.get('output_file', 'unknown')} ({headings_count} headings)")
            
            return summary
            
        except Exception as e:
            print(f"Critical error in process_all_pdfs: {e}")
            return {
                'status': 'critical_error',
                'message': str(e),
                'total_pdfs': 0,
                'successful': 0,
                'failed': 0
            }


def main():
    """
    Main entry point with improved processing
    """
    print("🎯 Adobe Round 1A: Improved PDF Hierarchy Detection")
    print("🔧 Advanced NLP + Selective Filtering")
    print("📊 Focus: Quality over Quantity")
    print("-" * 50)
    
    try:
        # Initialize improved processor
        processor = ImprovedRound1AProcessor()
        
        # Process all PDFs
        results = processor.process_all_pdfs()
        
        # Exit with appropriate code
        if results['status'] == 'completed' and results['failed'] == 0:
            print("\n🎉 All files processed successfully!")
            print("💡 Check the JSON files - they should be much cleaner now!")
            sys.exit(0)
        elif results['status'] == 'completed':
            print(f"\n⚠️  Processing completed with {results['failed']} failures")
            sys.exit(1)
        else:
            print(f"\n💥 Processing failed: {results.get('message', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Critical error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()