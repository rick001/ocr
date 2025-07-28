# 📄 Image/PDF to PDF OCR Converter

A powerful application that converts images and image-based PDFs into readable PDF documents using Optical Character Recognition (OCR) technology.

## ✨ Features

- **Web Interface**: User-friendly Streamlit web application
- **Command Line**: Simple CLI tool for batch processing
- **Multiple Formats**: Supports PNG, JPG, JPEG, BMP, TIFF, PDF
- **PDF Processing**: Handles both image-based PDFs and text-based PDFs
- **Advanced Processing**: Image preprocessing for better OCR accuracy
- **Clean PDF Output**: Properly formatted, readable PDF documents
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

### Prerequisites

1. **Python 3.7+** installed on your system
2. **Tesseract OCR** engine installed
3. **Poppler-utils** (for PDF processing on Linux/macOS)

#### Installing Tesseract OCR

**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Or use chocolatey:
choco install tesseract
```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

#### Installing Poppler-utils (for PDF processing)

**Windows:**
```bash
# Download from: http://blog.alivate.com.au/poppler-windows/
# Extract to a directory and add to PATH
```

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

### Installation

1. **Clone or download this repository**
2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Web Interface (Recommended)

Launch the web application:
```bash
streamlit run ocr_app.py
```

Then open your browser to `http://localhost:8501` and:
1. Upload an image or PDF containing text
2. Click "Convert to PDF"
3. Download the generated PDF

### Command Line Interface

Convert a single image:
```bash
python cli_ocr.py document.png
```

Convert a PDF:
```bash
python cli_ocr.py scanned_document.pdf
```

Specify output filename:
```bash
python cli_ocr.py image.jpg output.pdf
```

Enable verbose output:
```bash
python cli_ocr.py document.png --verbose
```

Get help:
```bash
python cli_ocr.py --help
```

### Batch Processing

Process all files in a directory:
```bash
python batch_ocr.py ./documents
```

Specify output directory:
```bash
python batch_ocr.py ./input_folder ./output_folder
```

## 🛠️ Technical Details

### Architecture

The application consists of several key components:

1. **File Analysis** (`OCRProcessor.is_pdf_image_based()`):
   - Determines if PDF contains text or images
   - Uses PyPDF2 for text extraction analysis

2. **PDF Processing** (`OCRProcessor.extract_text_from_pdf()`):
   - Converts PDF pages to images using pdf2image
   - Processes each page individually with OCR

3. **Image Preprocessing** (`OCRProcessor.preprocess_image()`):
   - Grayscale conversion
   - Noise reduction using OpenCV
   - Binary thresholding
   - Morphological operations

4. **OCR Processing** (`OCRProcessor.extract_text()`):
   - Tesseract OCR engine
   - Optimized configuration for text extraction
   - Error handling and validation

5. **PDF Generation** (`OCRProcessor.create_pdf()`):
   - ReportLab for PDF creation
   - Proper text formatting and styling
   - Paragraph separation and spacing

### Dependencies

- **pytesseract**: Python wrapper for Tesseract OCR
- **Pillow**: Image processing and manipulation
- **opencv-python**: Advanced image preprocessing
- **reportlab**: PDF generation and formatting
- **streamlit**: Web interface framework
- **numpy**: Numerical operations for image processing
- **pdf2image**: PDF to image conversion
- **PyPDF2**: PDF text extraction and analysis

## 📋 Supported Formats

### Input Formats
- **Images**: PNG, JPG, JPEG, BMP, TIFF
- **PDFs**: Image-based PDFs (scanned documents), text-based PDFs

### Output Format
- PDF (readable, searchable text)

## 🎯 Best Practices

For optimal OCR results:

1. **Image Quality**:
   - Use high-resolution images (300+ DPI)
   - Ensure good lighting and contrast
   - Avoid blurry or distorted text

2. **Text Clarity**:
   - Clear, well-spaced text
   - Good contrast between text and background
   - Avoid handwritten text (for best results)

3. **File Preparation**:
   - Crop unnecessary areas
   - Ensure text is properly oriented
   - Remove watermarks or overlays if possible

4. **PDF Processing**:
   - The app automatically detects PDF type
   - Image-based PDFs are processed page by page
   - Text-based PDFs are extracted directly

## 🔧 Troubleshooting

### Common Issues

**"Tesseract not found" error:**
- Ensure Tesseract is installed and in your system PATH
- On Windows, you may need to set the path manually in the code

**"Poppler not found" error (PDF processing):**
- Install poppler-utils on your system
- Windows users need to download and configure poppler manually

**Poor OCR accuracy:**
- Try preprocessing the image manually (increase contrast, remove noise)
- Ensure the image has sufficient resolution
- Check that text is clearly visible and well-spaced

**PDF generation fails:**
- Check that the output directory is writable
- Ensure sufficient disk space
- Verify that extracted text is not empty

### Performance Tips

- For batch processing, use the CLI version
- Large images and multi-page PDFs may take longer to process
- Consider resizing very large images before processing
- PDF processing is more resource-intensive than image processing

## 📁 Project Structure

```
ocr/
├── ocr_app.py          # Main web application
├── cli_ocr.py          # Command-line interface
├── batch_ocr.py        # Batch processing tool
├── test_installation.py # Installation verification
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Tesseract OCR**: The powerful OCR engine that makes this possible
- **OpenCV**: For advanced image processing capabilities
- **Streamlit**: For the beautiful web interface
- **ReportLab**: For professional PDF generation
- **pdf2image**: For PDF to image conversion
- **PyPDF2**: For PDF text analysis

---

**Happy converting! 📄✨** 