#!/usr/bin/env python3
"""
Command-line OCR tool for converting images and PDFs to readable PDFs.
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
        description="Convert images and PDFs to readable PDFs using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_ocr.py document.png
  python cli_ocr.py scanned_document.pdf
  python cli_ocr.py image.jpg output.pdf
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
    
    # Initialize OCR processor
    processor = OCRProcessor()
    
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
            
        else:
            print("❌ Failed to create PDF. Please check the input file.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 