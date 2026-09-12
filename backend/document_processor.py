import json
from pathlib import Path
from typing import Dict, Any, List
import pymupdf  # PyMuPDF

from backend.text_cleaner import clean_extracted_text
from backend.chunker import create_chunks_from_pages
from backend.config import EXTRACTED_DIR, CHUNKS_DIR

class DocumentProcessor:
    @staticmethod
    def extract_and_chunk_pdf(pdf_path: Path, original_filename: str) -> Dict[str, Any]:
        """
        Extracts text page-by-page from a PDF, cleans it, chunks it,
        and saves intermediate artifacts.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            raise ValueError(f"Could not open PDF file. It may be corrupted or invalid: {e}")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError("The uploaded PDF has 0 pages.")

        pages_data: List[Dict[str, Any]] = []
        total_extracted_characters = 0

        for page_index in range(total_pages):
            page_num = page_index + 1
            page = doc[page_index]
            raw_text = page.get_text("text") or ""
            cleaned_text = clean_extracted_text(raw_text)

            total_extracted_characters += len(cleaned_text)
            pages_data.append({
                "page": page_num,
                "text": cleaned_text
            })

        doc.close()

        if total_extracted_characters < 50:
            raise ValueError(
                "PDF contains almost no extractable text. It may be a scanned image-only PDF. "
                "Please upload a text-based educational textbook PDF."
            )

        # Generate chunks
        chunks = create_chunks_from_pages(pages_data, original_filename)

        if not chunks:
            raise ValueError("Failed to create text chunks from the extracted PDF text.")

        # Save extracted pages artifact
        safe_stem = Path(original_filename).stem.replace(" ", "_")
        extracted_file = EXTRACTED_DIR / f"{safe_stem}_extracted.json"
        with open(extracted_file, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)

        # Save chunks artifact
        chunks_file = CHUNKS_DIR / f"{safe_stem}_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return {
            "filename": original_filename,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "pages_data": pages_data,
            "chunks": chunks
        }
