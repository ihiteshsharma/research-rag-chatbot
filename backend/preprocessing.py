import re

def clean_text(text):
    """Clean extracted text."""
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"Figure \d+", "", text)
    text = re.sub(r"Table \d+", "", text)
    return text

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks