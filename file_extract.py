import io
import pandas as pd
from docx import Document
import pdfplumber
from PyPDF2 import PdfReader

def extract_text_from_file(uploaded_file):
    """
    Extracts readable text from various file types.
    Supports txt, csv, docx, pdf.
    """
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()

    file_bytes = uploaded_file.read()
    text = ""

    try:
        if ext == "txt":
            text = file_bytes.decode("utf-8", errors="ignore")

        elif ext == "csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
            text = df.to_string()

        elif ext in ("docx", "doc"):
            with open("temp_docx.docx", "wb") as tmp:
                tmp.write(file_bytes)
            doc = Document("temp_docx.docx")
            text = "\n".join(p.text for p in doc.paragraphs)

        elif ext == "pdf":
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    pages = []
                    for p in pdf.pages:
                        t = p.extract_text()
                        if t:
                            pages.append(t)
                    text = "\n\n".join(pages)
            except Exception:
                reader = PdfReader(io.BytesIO(file_bytes))
                pages = []
                for p in reader.pages:
                    try:
                        t = p.extract_text()
                        if t:
                            pages.append(t)
                    except Exception:
                        continue
                text = "\n\n".join(pages)
        else:
            text = file_bytes.decode("utf-8", errors="ignore")

    except Exception as e:
        text = f"Error extracting text: {e}"

    return text
