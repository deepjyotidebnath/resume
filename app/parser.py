"""Extract plain text and basic layout signals from PDF / DOCX / TXT resumes."""
import io
from dataclasses import dataclass


@dataclass
class ParsedResume:
    text: str
    pages: int
    tables: int = 0
    images: int = 0


class UnsupportedFileError(ValueError):
    pass


def _parse_pdf(data: bytes) -> ParsedResume:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    chunks, images = [], 0
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
        try:
            images += len(page.images)
        except Exception:
            pass
    return ParsedResume(text="\n".join(chunks), pages=len(reader.pages), images=images)


def _parse_docx(data: bytes) -> ParsedResume:
    from docx import Document

    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                lines.append(cell.text)
    text = "\n".join(lines)
    pages = max(1, round(len(text.split()) / 500))  # DOCX has no fixed pages; estimate
    return ParsedResume(
        text=text, pages=pages, tables=len(doc.tables), images=len(doc.inline_shapes)
    )


def extract(filename: str, data: bytes) -> ParsedResume:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        return _parse_pdf(data)
    if ext == "docx":
        return _parse_docx(data)
    if ext == "txt":
        text = data.decode("utf-8", errors="ignore")
        return ParsedResume(text=text, pages=max(1, round(len(text.split()) / 500)))
    raise UnsupportedFileError("Upload a PDF, DOCX or TXT file.")
