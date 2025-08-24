import PyPDF2
import docx
import pandas as pd
import io
import re
from typing import Optional, Tuple
import streamlit as st

class DocumentProcessor:
    """
    Process various document formats to extract text for political leaning analysis.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.supported_formats = {
            'pdf': self.extract_from_pdf,
            'txt': self.extract_from_txt,
            'csv': self.extract_from_csv,
            'docx': self.extract_from_docx
        }
    
    def extract_from_pdf(self, file_content: bytes) -> Tuple[str, bool]:
        """
        Extract text from PDF file.
        
        Args:
            file_content: PDF file content as bytes
            
        Returns:
            Tuple of (extracted_text, success_status)
        """
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Clean up the text
            text = self.clean_text(text)
            return text, True
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return "", False
    
    def extract_from_txt(self, file_content: bytes) -> Tuple[str, bool]:
        """
        Extract text from TXT file.
        
        Args:
            file_content: TXT file content as bytes
            
        Returns:
            Tuple of (extracted_text, success_status)
        """
        try:
            text = file_content.decode('utf-8')
            text = self.clean_text(text)
            return text, True
            
        except Exception as e:
            st.error(f"Error extracting text from TXT file: {e}")
            return "", False
    
    def extract_from_csv(self, file_content: bytes) -> Tuple[str, bool]:
        """
        Extract text from CSV file.
        
        Args:
            file_content: CSV file content as bytes
            
        Returns:
            Tuple of (extracted_text, success_status)
        """
        try:
            # Try to read as CSV and extract text from 'text' column
            df = pd.read_csv(io.BytesIO(file_content))
            
            if 'text' in df.columns:
                # If there's a text column, combine all texts
                texts = df['text'].dropna().tolist()
                text = "\n\n".join(texts)
            else:
                # If no text column, combine all columns
                text = df.to_string()
            
            text = self.clean_text(text)
            return text, True
            
        except Exception as e:
            st.error(f"Error extracting text from CSV file: {e}")
            return "", False
    
    def extract_from_docx(self, file_content: bytes) -> Tuple[str, bool]:
        """
        Extract text from DOCX file.
        
        Args:
            file_content: DOCX file content as bytes
            
        Returns:
            Tuple of (extracted_text, success_status)
        """
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            text = self.clean_text(text)
            return text, True
            
        except Exception as e:
            st.error(f"Error extracting text from DOCX file: {e}")
            return "", False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([\.\,\!\?\;\:])', r'\1', text)
        
        return text.strip()
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[str, bool, str]:
        """
        Process an uploaded file and extract text.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (extracted_text, success_status, file_type)
        """
        if uploaded_file is None:
            return "", False, ""
        
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Check if format is supported
        if file_extension not in self.supported_formats:
            st.error(f"Unsupported file format: {file_extension}")
            st.info("Supported formats: PDF, TXT, CSV, DOCX")
            return "", False, file_extension
        
        # Extract text based on file type
        extractor = self.supported_formats[file_extension]
        text, success = extractor(uploaded_file.getvalue())
        
        return text, success, file_extension
    
    def get_supported_formats(self) -> list:
        """Get list of supported file formats."""
        return list(self.supported_formats.keys()) 