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

class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor with Tesseract configuration."""
        # Configure Tesseract for better OCR results
        self.custom_config = r'--oem 3 --psm 6'
        
    def preprocess_image(self, image):
        """
        Preprocess the image to improve OCR accuracy.
        
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
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
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
            
            return "\n\n".join(all_text)
            
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
                    return self.extract_text_from_pdf(str(file_path))
                else:
                    # PDF contains text, extract directly
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text_parts = []
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text.strip():
                                text_parts.append(text.strip())
                        return "\n\n".join(text_parts)
            else:
                # Handle as image file
                image = Image.open(file_path)
                
                # Preprocess the image
                processed_image = self.preprocess_image(image)
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(processed_image, config=self.custom_config)
                
                return text.strip()
            
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
    
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

def main():
    """Main function for the OCR application."""
    st.set_page_config(
        page_title="Image/PDF to PDF OCR Converter",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Image/PDF to PDF OCR Converter")
    st.markdown("Convert images and image-based PDFs into readable PDF documents using OCR technology.")
    
    # Initialize OCR processor
    processor = OCRProcessor()
    
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
            st.write("The PDF will be processed to extract text using OCR.")
        
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
    2. **Click 'Convert to PDF'** to process the file using OCR
    3. **Download the PDF** with the extracted text in a readable format
    
    **Supported file types:**
    - **Images**: PNG, JPG, JPEG, BMP, TIFF
    - **PDFs**: Image-based PDFs (scanned documents, etc.)
    
    **Tips for better results:**
    - Use clear, high-resolution images
    - Ensure good lighting and contrast
    - Avoid blurry or distorted text
    - Text should be clearly visible and well-spaced
    - For PDFs, the app will automatically detect if it's image-based or text-based
    """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Technologies used:**
        - **OCR Engine**: Tesseract (via pytesseract)
        - **Image Processing**: OpenCV and PIL
        - **PDF Processing**: pdf2image and PyPDF2
        - **PDF Generation**: ReportLab
        - **Web Interface**: Streamlit
        
        **Processing steps:**
        1. **File Analysis**: Determine if input is image or PDF
        2. **PDF Processing**: Convert PDF pages to images (if image-based)
        3. **Image Preprocessing**: Grayscale conversion, noise reduction, thresholding
        4. **Text Extraction**: Using Tesseract OCR
        5. **PDF Generation**: Create readable, searchable PDF
        6. **Download**: Provide processed file for download
        """)

if __name__ == "__main__":
    main() 