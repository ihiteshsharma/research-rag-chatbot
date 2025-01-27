import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
import io

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    return tables

def extract_images_from_pdf(pdf_path):
    """Extract images and perform OCR."""
    doc = fitz.open(pdf_path)
    extracted_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            extracted_text += pytesseract.image_to_string(image)
    return extracted_text


def process_pdf(pdf_path):
    """Process a single PDF to extract text, tables, and images."""
    text = extract_text_from_pdf(pdf_path)
    tables = extract_tables_from_pdf(pdf_path)
    image_text = extract_images_from_pdf(pdf_path)
    return text + image_text, tables