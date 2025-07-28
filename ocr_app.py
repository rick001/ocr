import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import streamlit as st
import tempfile
from pathlib import Path
from pdf2image import convert_from_path
import PyPDF2
from llm_enhancer import LLMEnhancer

class OCRProcessor:
    def __init__(self, use_llm: bool = True):
        """Initialize the OCR processor with Tesseract configuration."""
        # Configure Tesseract for better OCR results focused on document content
        self.custom_config = r'--oem 3 --psm 6'
        
        # Initialize LLM enhancer if requested
        self.use_llm = use_llm
        self.llm_enhancer = None
        if use_llm:
            try:
                self.llm_enhancer = LLMEnhancer()
                st.success("✅ LLM enhancement enabled with DeepSeek model")
            except Exception as e:
                st.warning(f"⚠️ LLM enhancement disabled: {str(e)}")
                self.use_llm = False
        
    def preprocess_image(self, image):
        """
        Preprocess the image to improve OCR accuracy and focus on document content.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding for better text separation
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Additional cleaning to preserve text structure
        kernel_vertical = np.ones((2, 1), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_vertical)
        
        # Focus on document content by removing UI elements
        cleaned = self._remove_ui_elements(cleaned)
        
        return cleaned
    
    def _remove_ui_elements(self, image):
        """
        Remove UI elements like sidebars, headers, and navigation areas.
        
        Args:
            image: Binary image
            
        Returns:
            Image with UI elements removed
        """
        # Get image dimensions
        height, width = image.shape
        
        # Define regions to potentially remove (UI elements typically appear in these areas)
        # Top header area (usually contains navigation, buttons, etc.)
        top_crop_height = int(height * 0.15)  # Remove top 15%
        
        # Sidebar areas (left and right)
        left_crop_width = int(width * 0.2)   # Remove left 20%
        right_crop_width = int(width * 0.1)  # Remove right 10%
        
        # Bottom footer area
        bottom_crop_height = int(height * 0.1)  # Remove bottom 10%
        
        # Crop to focus on main content area
        cropped = image[top_crop_height:height-bottom_crop_height, 
                       left_crop_width:width-right_crop_width]
        
        # If the cropped area is too small, use the original
        if cropped.shape[0] < height * 0.5 or cropped.shape[1] < width * 0.5:
            return image
        
        return cropped
    
    def _detect_document_region(self, image):
        """
        Detect the main document region using contour detection.
        
        Args:
            image: Binary image
            
        Returns:
            Cropped image focusing on document content
        """
        # Find contours in the image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (likely the main document area)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding around the detected region
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop to the detected document region
        cropped = image[y:y+h, x:x+w]
        
        return cropped
    
    def is_pdf_image_based(self, pdf_path):
        """
        Check if a PDF contains image-based content rather than text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF is image-based, False if it contains text
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF has any text content
                has_text = False
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        has_text = True
                        break
                
                # If no text found, assume it's image-based
                return not has_text
                
        except Exception as e:
            st.warning(f"Could not analyze PDF structure: {str(e)}")
            # Default to treating as image-based if analysis fails
            return True
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF by converting pages to images and using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            # Try using pdf2image first (requires Poppler)
            try:
                # Convert PDF pages to images
                images = convert_from_path(pdf_path, dpi=300)
                
                all_text = []
                
                for i, image in enumerate(images):
                    # Preprocess the image
                    processed_image = self.preprocess_image(image)
                    
                    # Extract text from the page
                    page_text = pytesseract.image_to_string(processed_image, config=self.custom_config)
                    
                    if page_text.strip():
                        all_text.append(f"--- Page {i+1} ---\n{page_text.strip()}")
                
                result = "\n\n".join(all_text)
                return result
                
            except Exception as e:
                st.warning(f"pdf2image failed: {str(e)}. Trying PyMuPDF...")
                
                # Fallback: Try using PyMuPDF if available
                try:
                    import fitz  # PyMuPDF
                    
                    doc = fitz.open(pdf_path)
                    all_text = []
                    
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        
                        # Convert page to image
                        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Preprocess the image
                        processed_image = self.preprocess_image(image)
                        
                        # Extract text from the page
                        page_text = pytesseract.image_to_string(processed_image, config=self.custom_config)
                        
                        if page_text.strip():
                            all_text.append(f"--- Page {page_num+1} ---\n{page_text.strip()}")
                    
                    doc.close()
                    result = "\n\n".join(all_text)
                    return result
                    
                except ImportError:
                    st.error("PyMuPDF not available. Please install it with: pip install PyMuPDF")
                    return ""
                except Exception as e2:
                    st.error(f"PyMuPDF processing failed: {str(e2)}")
                    return ""
                
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_text(self, file_path):
        """
        Extract text from an image or PDF file using OCR.
        
        Args:
            file_path: Path to the image or PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            file_path = Path(file_path)
            
            # Check if it's a PDF
            if file_path.suffix.lower() == '.pdf':
                # Check if PDF is image-based
                if self.is_pdf_image_based(str(file_path)):
                    raw_text = self.extract_text_from_pdf(str(file_path))
                else:
                    # PDF contains text, extract directly
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text_parts = []
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text.strip():
                                text_parts.append(text.strip())
                        raw_text = "\n\n".join(text_parts)
            else:
                # Handle as image file
                image = Image.open(file_path)
                
                # Preprocess the image
                processed_image = self.preprocess_image(image)
                
                # Try multiple OCR configurations for better accuracy
                raw_text = self._extract_text_with_multiple_configs(processed_image)
            
            # Enhance text using LLM if available
            if self.use_llm and self.llm_enhancer and raw_text.strip():
                with st.spinner("🤖 Enhancing text with AI..."):
                    enhanced_text = self.llm_enhancer.enhance_ocr_text(raw_text.strip())
                    return enhanced_text
            
            return raw_text.strip()
            
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
    
    def _extract_text_with_multiple_configs(self, image):
        """
        Extract text using multiple OCR configurations for better accuracy.
        Focused on document content extraction.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Best extracted text
        """
        # Configurations optimized for document content
        configs = [
            r'--oem 3 --psm 6',   # Uniform block of text (best for documents)
            r'--oem 3 --psm 3',   # Fully automatic page segmentation
            r'--oem 3 --psm 4',   # Assume a single column of text
            r'--oem 3 --psm 1',   # Automatic page segmentation with OSD
            r'--oem 3 --psm 12'   # Sparse text with OSD
        ]
        
        best_text = ""
        best_confidence = 0
        
        for config in configs:
            try:
                # Extract text with confidence scores
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Extract text
                text = pytesseract.image_to_string(image, config=config)
                
                # Clean up the text to focus on document content
                cleaned_text = self._clean_extracted_text(text)
                
                # Choose the result with highest confidence and good content
                if (avg_confidence > best_confidence and 
                    cleaned_text.strip() and 
                    len(cleaned_text.strip()) > len(best_text.strip())):
                    best_confidence = avg_confidence
                    best_text = cleaned_text
                    
            except Exception as e:
                st.warning(f"OCR config failed: {str(e)}")
                continue
        
        # If no good result found, use default config
        if not best_text.strip():
            text = pytesseract.image_to_string(image, config=self.custom_config)
            best_text = self._clean_extracted_text(text)
        
        return best_text
    
    def _clean_extracted_text(self, text):
        """
        Clean extracted text to focus on document content.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text focused on document content
        """
        if not text:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip lines that are likely UI elements
            if self._is_ui_element(line):
                continue
            
            # Skip very short lines that might be noise
            if len(line) < 3:
                continue
            
            # Skip lines that are mostly special characters
            if self._is_noise_line(line):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_ui_element(self, line):
        """
        Check if a line is likely a UI element.
        
        Args:
            line: Text line to check
            
        Returns:
            True if line appears to be a UI element
        """
        line_lower = line.lower()
        
        # Common UI element indicators
        ui_indicators = [
            'file', 'edit', 'view', 'tools', 'help',  # Menu items
            'share', 'comment', 'star', 'request',     # Google Docs buttons
            'document tabs', 'page', 'of',             # Navigation
            'search', 'find', 'replace',               # Search tools
            'zoom', 'fit', 'actual',                   # Zoom controls
            'undo', 'redo', 'copy', 'paste',           # Edit tools
            'bold', 'italic', 'underline',             # Formatting
            'align', 'left', 'center', 'right',        # Alignment
            'font', 'size', 'color',                   # Text formatting
            'insert', 'table', 'image', 'link',        # Insert menu
            'format', 'paragraph', 'list',             # Format menu
            'tools', 'spelling', 'grammar',            # Tools menu
            'add-ons', 'extensions',                   # Add-ons
            'account', 'profile', 'settings',          # Account/settings
            'save', 'download', 'print',               # File operations
            'new', 'open', 'close',                    # File menu
            'recent', 'starred', 'shared',             # File organization
        ]
        
        # Check if line contains UI indicators
        for indicator in ui_indicators:
            if indicator in line_lower:
                return True
        
        # Check for patterns typical of UI elements
        ui_patterns = [
            r'^\d+$',                    # Just numbers (page numbers)
            r'^[A-Z\s]+$',              # All caps (menu items)
            r'^[^\w\s]+$',              # Only special characters
            r'^\s*[•·]\s*$',            # Just bullet points
            r'^\s*[-_=]+\s*$',          # Just separators
        ]
        
        import re
        for pattern in ui_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _is_noise_line(self, line):
        """
        Check if a line is noise (not meaningful content).
        
        Args:
            line: Text line to check
            
        Returns:
            True if line appears to be noise
        """
        if not line:
            return True
        
        # Count different character types
        letters = sum(1 for c in line if c.isalpha())
        digits = sum(1 for c in line if c.isdigit())
        spaces = sum(1 for c in line if c.isspace())
        special = len(line) - letters - digits - spaces
        
        # If mostly special characters, it's likely noise
        if special > len(line) * 0.7:
            return True
        
        # If very short and mostly numbers/special chars
        if len(line) < 5 and (digits + special) > letters:
            return True
        
        return False
    
    def create_pdf(self, text, output_path):
        """
        Create a readable PDF from extracted text.
        
        Args:
            text: Extracted text content
            output_path: Path where PDF will be saved
            
        Returns:
            Path to the created PDF file
        """
        try:
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Create custom style for better readability
            custom_style = ParagraphStyle(
                'CustomStyle',
                parent=styles['Normal'],
                fontSize=12,
                leading=16,
                spaceAfter=6
            )
            
            # Prepare content
            content = []
            
            # Split text into paragraphs
            paragraphs = text.split('\n\n')
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Clean up the paragraph
                    clean_paragraph = paragraph.strip().replace('\n', ' ')
                    if clean_paragraph:
                        content.append(Paragraph(clean_paragraph, custom_style))
                        content.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(content)
            
            return output_path
            
        except Exception as e:
            st.error(f"Error creating PDF: {str(e)}")
            return None
    
    def process_file_to_pdf(self, file_path, output_path=None):
        """
        Complete pipeline: image or PDF to readable PDF.
        
        Args:
            file_path: Path to input image or PDF file
            output_path: Path for output PDF (optional)
            
        Returns:
            Path to the created PDF file
        """
        if output_path is None:
            # Generate output path
            input_name = Path(file_path).stem
            output_path = f"{input_name}_ocr.pdf"
        
        # Extract text from file
        text = self.extract_text(file_path)
        
        if not text:
            st.warning("No text was extracted from the file.")
            return None
        
        # Create PDF from extracted text
        pdf_path = self.create_pdf(text, output_path)
        
        return pdf_path
    
    def extract_structured_data(self, text, data_type="general"):
        """
        Extract structured data from text using LLM.
        
        Args:
            text: Text to analyze
            data_type: Type of data to extract
            
        Returns:
            Dictionary with extracted structured data
        """
        if self.use_llm and self.llm_enhancer:
            return self.llm_enhancer.extract_structured_data(text, data_type)
        else:
            return {
                "extracted_text": text,
                "confidence": "low",
                "error": "LLM enhancement not available"
            }

def main():
    """Main function for the OCR application."""
    st.set_page_config(
        page_title="AI-Enhanced OCR Converter",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI-Enhanced OCR Converter")
    st.markdown("Convert images and PDFs into readable documents using OCR + AI enhancement.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM enhancement toggle
        use_llm = st.checkbox("Enable AI Enhancement", value=True, 
                             help="Use DeepSeek AI model to improve OCR results")
        
        # Document type selection
        doc_type = st.selectbox(
            "Document Type",
            ["general", "receipt", "invoice", "form", "letter"],
            help="Helps AI better understand and process the document"
        )
        
        # Show API status
        if use_llm:
            try:
                test_enhancer = LLMEnhancer()
                st.success("✅ AI Enhancement Ready")
            except Exception as e:
                st.error(f"❌ AI Enhancement Error: {str(e)}")
                st.info("💡 Set OPENROUTER_API_KEY environment variable")
    
    # Initialize OCR processor
    processor = OCRProcessor(use_llm=use_llm)
    
    # File upload section
    st.header("📁 Upload File")
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pdf'],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, PDF"
    )
    
    if uploaded_file is not None:
        # Display uploaded file info
        st.subheader("📸 Uploaded File")
        
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        if file_type.startswith('image/'):
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {file_name}", use_column_width=True)
        elif file_type == 'application/pdf':
            st.info(f"📄 PDF File: {file_name}")
            st.write("The PDF will be processed to extract text using OCR + AI enhancement.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Process button
        if st.button("🔄 Convert to PDF", type="primary"):
            with st.spinner("Processing file..."):
                # Process the file
                pdf_path = processor.process_file_to_pdf(temp_file_path)
                
                if pdf_path and os.path.exists(pdf_path):
                    st.success("✅ PDF created successfully!")
                    
                    # Display download button
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf"
                        )
                    
                    # Show structured data extraction if LLM is available
                    if use_llm and processor.llm_enhancer:
                        st.subheader("🔍 AI Analysis")
                        
                        # Extract text for analysis
                        text = processor.extract_text(temp_file_path)
                        
                        if text:
                            # Extract structured data
                            structured_data = processor.extract_structured_data(text, doc_type)
                            
                            if structured_data and "error" not in structured_data:
                                st.json(structured_data)
                            else:
                                st.warning("Could not extract structured data")
                    
                    # Clean up temporary files
                    try:
                        os.unlink(temp_file_path)
                        os.unlink(pdf_path)
                    except:
                        pass
                else:
                    st.error("❌ Failed to create PDF. Please try again with a different file.")
    
    # Instructions section
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. **Upload an image or PDF** containing text that you want to convert to readable PDF
    2. **Configure AI enhancement** in the sidebar (requires OpenRouter API key)
    3. **Click 'Convert to PDF'** to process the file using OCR + AI
    4. **Download the PDF** with enhanced, readable text
    5. **View AI analysis** for structured data extraction
    
    **AI Enhancement Features:**
    - Fixes OCR errors and misinterpretations
    - Improves text formatting and readability
    - Extracts structured data (receipts, invoices, etc.)
    - Provides better text interpretation
    """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Technologies used:**
        - **OCR Engine**: Tesseract (via pytesseract)
        - **AI Enhancement**: DeepSeek Chat v3 via OpenRouter.ai
        - **Image Processing**: OpenCV and PIL
        - **PDF Processing**: pdf2image and PyPDF2
        - **PDF Generation**: ReportLab
        - **Web Interface**: Streamlit
        
        **Processing steps:**
        1. **File Analysis**: Determine if input is image or PDF
        2. **PDF Processing**: Convert PDF pages to images (if image-based)
        3. **Image Preprocessing**: Grayscale conversion, noise reduction, thresholding
        4. **Text Extraction**: Using Tesseract OCR
        5. **AI Enhancement**: DeepSeek model improves text quality
        6. **PDF Generation**: Create readable, searchable PDF
        7. **Structured Data**: AI extracts key information
        8. **Download**: Provide processed file for download
        """)

if __name__ == "__main__":
    main() 