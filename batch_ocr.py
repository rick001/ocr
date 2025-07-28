#!/usr/bin/env python3
"""
Batch OCR processor for converting multiple images and PDFs to PDFs.
Usage: python batch_ocr.py <input_directory> [output_directory]
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ocr_app import OCRProcessor

def process_single_file(args):
    """Process a single image or PDF file."""
    file_path, output_dir, processor = args
    
    try:
        # Generate output filename
        input_name = Path(file_path).stem
        output_path = output_dir / f"{input_name}_ocr.pdf"
        
        # Process the file
        pdf_path = processor.process_file_to_pdf(str(file_path), str(output_path))
        
        if pdf_path and os.path.exists(pdf_path):
            return (file_path, pdf_path, True, None)
        else:
            return (file_path, None, False, "Failed to create PDF")
            
    except Exception as e:
        return (file_path, None, False, str(e))

def main():
    """Batch process multiple images and PDFs to PDFs."""
    parser = argparse.ArgumentParser(
        description="Batch convert images and PDFs to readable PDFs using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_ocr.py ./documents
  python batch_ocr.py ./input_folder ./output_folder
  python batch_ocr.py --help
        """
    )
    
    parser.add_argument(
        'input_directory',
        help='Directory containing input images and PDFs'
    )
    
    parser.add_argument(
        'output_directory',
        nargs='?',
        help='Directory for output PDFs (defaults to input directory)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of worker threads (default: 4)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_directory)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"❌ Error: Input directory '{args.input_directory}' not found or not a directory.")
        sys.exit(1)
    
    # Set output directory
    if args.output_directory:
        output_dir = Path(args.output_directory)
    else:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all supported files
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']
    files_to_process = []
    
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            files_to_process.append(file_path)
    
    if not files_to_process:
        print(f"❌ No supported files found in '{input_dir}'")
        print(f"Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)
    
    print(f"📁 Found {len(files_to_process)} files to process")
    print(f"📄 Output directory: {output_dir}")
    
    # Count file types
    file_types = {}
    for file_path in files_to_process:
        ext = file_path.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    if args.verbose:
        print("\n📋 Files to process:")
        for file_path in files_to_process:
            print(f"  - {file_path.name}")
        print(f"\n📊 File type breakdown:")
        for ext, count in file_types.items():
            print(f"  - {ext.upper()}: {count} files")
        print()
    
    # Initialize OCR processor
    processor = OCRProcessor()
    
    # Process files in parallel
    print(f"🔄 Processing {len(files_to_process)} files with {args.workers} workers...")
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Prepare arguments for each task
        tasks = [(str(file_path), output_dir, processor) for file_path in files_to_process]
        
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, task): task[0] for task in tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path, pdf_path, success, error = future.result()
            
            if success:
                successful += 1
                if args.verbose:
                    print(f"✅ {Path(file_path).name} → {Path(pdf_path).name}")
                else:
                    print(f"✅ {Path(file_path).name}")
            else:
                failed += 1
                print(f"❌ {Path(file_path).name}: {error}")
    
    # Summary
    print(f"\n📊 Processing complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📄 PDFs saved to: {output_dir}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main() 