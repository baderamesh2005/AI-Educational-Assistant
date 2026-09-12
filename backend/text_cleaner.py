import re
import unicodedata

def clean_extracted_text(text: str) -> str:
    """
    Cleans raw text extracted from educational PDFs while preserving
    mathematical symbols, equations, and paragraph structure.
    """
    if not text:
        return ""

    # Normalize unicode characters (NFKC handles ligatures like ? -> fi, ? -> fl)
    text = unicodedata.normalize("NFKC", text)

    # Rejoin hyphenated words split across lines (e.g., "per- \n iodic" -> "periodic")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace multiple spaces or tabs with a single space, but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)

    # Clean up repeated blank lines (keep at most 2 newlines for paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove isolated non-printable characters except standard whitespace
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")

    return text.strip()
