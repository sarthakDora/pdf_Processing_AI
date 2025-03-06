import pdfplumber
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path


# Function to extract text from a digital PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip() 

# Function to perform OCR on a scanned PDF
def extract_text_from_scanned_pdf(pdf_path):
    images = convert_from_path(pdf_path)  # Convert PDF to images
    text = ""
    
    for img in images:
        # Convert to OpenCV format
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        text += pytesseract.image_to_string(gray) + "\n"

    return text.strip()

# Detect if PDF is digital or scanned
def process_pdf(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)

    if not extracted_text:
        print("No text found, now trying to extract by running OCR...")
        extracted_text = extract_text_from_scanned_pdf(pdf_path)

    return extracted_text


if __name__ == "__main__":
    pdf_file = "pdfs/Sample2.pdf" 
    extracted_text = process_pdf(pdf_file)
    print("Extracted Text:\n", extracted_text)