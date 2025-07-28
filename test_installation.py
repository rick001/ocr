#!/usr/bin/env python3
"""
Test script to verify OCR installation and functionality.
Run this script to check if all dependencies are properly installed.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import pytesseract
        print("✅ pytesseract imported successfully")
    except ImportError as e:
        print(f"❌ pytesseract import failed: {e}")
        return False
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        print("✅ ReportLab imported successfully")
    except ImportError as e:
        print(f"❌ ReportLab import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from pdf2image import convert_from_path
        print("✅ pdf2image imported successfully")
    except ImportError as e:
        print(f"❌ pdf2image import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    return True

def test_tesseract():
    """Test if Tesseract OCR is properly installed and accessible."""
    print("\n🔍 Testing Tesseract OCR...")
    
    try:
        import pytesseract
        
        # Test if tesseract is available
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        # Test basic functionality
        from PIL import Image
        import numpy as np
        
        # Create a simple test image with text
        test_image = Image.new('RGB', (200, 50), color='white')
        test_image.save('test_image.png')
        
        # Try to extract text (should work even if no text found)
        text = pytesseract.image_to_string(test_image)
        print("✅ Tesseract OCR functionality test passed")
        
        # Clean up test file
        if os.path.exists('test_image.png'):
            os.remove('test_image.png')
            
        return True
        
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure Tesseract is installed on your system")
        print("2. On Windows, you may need to set the Tesseract path manually")
        print("3. Try running: tesseract --version in your terminal")
        return False

def test_ocr_processor():
    """Test the OCR processor class."""
    print("\n🔍 Testing OCR processor...")
    
    try:
        from ocr_app import OCRProcessor
        
        # Initialize processor
        processor = OCRProcessor()
        print("✅ OCRProcessor initialized successfully")
        
        # Test image preprocessing
        import numpy as np
        test_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        processed = processor.preprocess_image(test_array)
        print("✅ Image preprocessing test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR processor test failed: {e}")
        return False

def test_pdf_generation():
    """Test PDF generation functionality."""
    print("\n🔍 Testing PDF generation...")
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        
        # Create a simple test PDF
        test_pdf = "test_output.pdf"
        c = canvas.Canvas(test_pdf, pagesize=A4)
        c.drawString(100, 750, "Test PDF Generation")
        c.save()
        
        if os.path.exists(test_pdf):
            print("✅ PDF generation test passed")
            os.remove(test_pdf)  # Clean up
            return True
        else:
            print("❌ PDF file was not created")
            return False
            
    except Exception as e:
        print(f"❌ PDF generation test failed: {e}")
        return False

def test_pdf_processing():
    """Test PDF processing capabilities."""
    print("\n🔍 Testing PDF processing...")
    
    try:
        from pdf2image import convert_from_path
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        
        # Create a simple test PDF
        test_pdf = "test_pdf_input.pdf"
        c = canvas.Canvas(test_pdf, pagesize=A4)
        c.drawString(100, 750, "Test PDF for Processing")
        c.save()
        
        # Test PDF to image conversion
        images = convert_from_path(test_pdf, dpi=150)
        print(f"✅ PDF to image conversion: {len(images)} page(s)")
        
        # Clean up test files
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        print("\n💡 Note: pdf2image requires poppler-utils on Linux/macOS")
        print("   - Ubuntu/Debian: sudo apt-get install poppler-utils")
        print("   - macOS: brew install poppler")
        print("   - Windows: Download from http://blog.alivate.com.au/poppler-windows/")
        return False

def test_pdf_analysis():
    """Test PDF text analysis capabilities."""
    print("\n🔍 Testing PDF text analysis...")
    
    try:
        import PyPDF2
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        
        # Create a simple test PDF with text
        test_pdf = "test_text_pdf.pdf"
        c = canvas.Canvas(test_pdf, pagesize=A4)
        c.drawString(100, 750, "Test PDF Text Analysis")
        c.save()
        
        # Test PDF text extraction
        with open(test_pdf, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = pdf_reader.pages[0].extract_text()
            print(f"✅ PDF text extraction: '{text.strip()}'")
        
        # Clean up test file
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
        
        return True
        
    except Exception as e:
        print(f"❌ PDF text analysis test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 OCR Application Installation Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test Tesseract
    if not test_tesseract():
        all_tests_passed = False
    
    # Test OCR processor
    if not test_ocr_processor():
        all_tests_passed = False
    
    # Test PDF generation
    if not test_pdf_generation():
        all_tests_passed = False
    
    # Test PDF processing
    if not test_pdf_processing():
        all_tests_passed = False
    
    # Test PDF text analysis
    if not test_pdf_analysis():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("🎉 All tests passed! Your OCR application is ready to use.")
        print("\n📖 Next steps:")
        print("1. Run the web interface: streamlit run ocr_app.py")
        print("2. Or use the CLI: python cli_ocr.py <file>")
        print("3. For batch processing: python batch_ocr.py <directory>")
        print("\n✨ New features:")
        print("- Support for image-based PDFs")
        print("- Automatic detection of PDF type (text vs image-based)")
        print("- Multi-page PDF processing")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\n🔧 Common solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Install Tesseract OCR on your system")
        print("3. Install poppler-utils for PDF processing (Linux/macOS)")
        print("4. Check the README.md for detailed installation instructions")
        sys.exit(1)

if __name__ == "__main__":
    main() 