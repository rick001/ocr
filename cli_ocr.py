#!/usr/bin/env python3
"""
Command-line OCR tool for converting images and PDFs to readable PDFs with AI enhancement.
Usage: python cli_ocr.py <input_file> [output_pdf]
"""

import sys
import os
import argparse
from pathlib import Path
from ocr_app import OCRProcessor

def main():
    """Command-line interface for the OCR application."""
    parser = argparse.ArgumentParser(
        description="Convert images and PDFs to readable PDFs using OCR + AI enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_ocr.py document.png
  python cli_ocr.py scanned_document.pdf
  python cli_ocr.py image.jpg output.pdf --ai
  python cli_ocr.py receipt.png --ai --doc-type receipt
  python cli_ocr.py --help
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the input image or PDF file'
    )
    
    parser.add_argument(
        'output_pdf',
        nargs='?',
        help='Path for the output PDF file (optional)'
    )
    
    parser.add_argument(
        '--ai', '--enhance',
        action='store_true',
        help='Enable AI enhancement using DeepSeek model'
    )
    
    parser.add_argument(
        '--doc-type',
        choices=['general', 'receipt', 'invoice', 'form', 'letter'],
        default='general',
        help='Document type for better AI processing (default: general)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Validate input file
    input_path = Path(args.input_file)
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']
    if not input_path.suffix.lower() in supported_formats:
        print(f"❌ Error: Unsupported file format '{input_path.suffix}'")
        print(f"Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)
    
    # Generate output path if not provided
    if args.output_pdf is None:
        output_path = input_path.with_suffix('.pdf')
        output_path = output_path.with_name(f"{input_path.stem}_ocr.pdf")
    else:
        output_path = Path(args.output_pdf)
    
    if args.verbose:
        print(f"📁 Input file: {args.input_file}")
        print(f"📄 Output PDF: {output_path}")
        print(f"🔍 File type: {input_path.suffix.upper()}")
        print(f"🤖 AI Enhancement: {'Enabled' if args.ai else 'Disabled'}")
        print(f"📋 Document type: {args.doc_type}")
    
    # Initialize OCR processor
    try:
        processor = OCRProcessor(use_llm=args.ai)
    except Exception as e:
        if args.ai:
            print(f"❌ AI enhancement error: {str(e)}")
            print("💡 Make sure OPENROUTER_API_KEY is set in environment variables")
            print("   Get your API key from: https://openrouter.ai/keys")
            sys.exit(1)
        else:
            processor = OCRProcessor(use_llm=False)
    
    print("🔄 Processing file...")
    
    try:
        # Process the file
        pdf_path = processor.process_file_to_pdf(args.input_file, str(output_path))
        
        if pdf_path and os.path.exists(pdf_path):
            print("✅ PDF created successfully!")
            print(f"📄 Output file: {pdf_path}")
            
            # Show file size
            file_size = os.path.getsize(pdf_path)
            print(f"📊 File size: {file_size:,} bytes")
            
            # Extract structured data if AI is enabled
            if args.ai and processor.llm_enhancer:
                print("🔍 Extracting structured data...")
                text = processor.extract_text(args.input_file)
                if text:
                    structured_data = processor.extract_structured_data(text, args.doc_type)
                    if structured_data and "error" not in structured_data:
                        print("📋 Structured data extracted:")
                        import json
                        print(json.dumps(structured_data, indent=2))
                    else:
                        print("⚠️ Could not extract structured data")
            
        else:
            print("❌ Failed to create PDF. Please check the input file.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 