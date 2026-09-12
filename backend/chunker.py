from typing import List, Dict, Any
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

def recursive_split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Recursively splits text into chunks of maximum size `chunk_size`
    with `chunk_overlap` using natural paragraph, sentence, and word boundaries.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
    
    def _split(txt: str, seps: List[str]) -> List[str]:
        if not txt:
            return []
        if len(txt) <= chunk_size:
            return [txt]
        if not seps:
            # Fallback to hard character slicing
            return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size - chunk_overlap)]

        sep = seps[0]
        remaining_seps = seps[1:]
        
        if sep == "":
            return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size - chunk_overlap)]

        parts = txt.split(sep)
        chunks = []
        current_chunk = ""

        for part in parts:
            candidate = current_chunk + (sep if current_chunk else "") + part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(part) > chunk_size:
                    # Sub-split long part with remaining separators
                    sub_chunks = _split(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
                    
        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap between consecutive chunks
        merged_with_overlap = []
        for idx, ch in enumerate(chunks):
            ch_clean = ch.strip()
            if not ch_clean:
                continue
            if idx > 0 and chunk_overlap > 0:
                prev_overlap = chunks[idx-1][-chunk_overlap:]
                combined = prev_overlap + " " + ch_clean
                merged_with_overlap.append(combined)
            else:
                merged_with_overlap.append(ch_clean)
                
        return merged_with_overlap

    return _split(text, separators)


def create_chunks_from_pages(pages_data: List[Dict[str, Any]], document_name: str) -> List[Dict[str, Any]]:
    """
    Takes extracted page dictionaries and splits each page into chunks
    while preserving page numbers and attaching structured metadata.
    """
    all_chunks = []
    chunk_counter = 1

    for page_item in pages_data:
        page_num = page_item.get("page", 1)
        raw_text = page_item.get("text", "").strip()

        if not raw_text:
            continue

        page_chunks = recursive_split_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_text in page_chunks:
            if len(chunk_text.strip()) < 20:
                # Skip tiny noise fragments
                continue

            all_chunks.append({
                "chunk_id": chunk_counter,
                "page": page_num,
                "document_name": document_name,
                "text": chunk_text.strip()
            })
            chunk_counter += 1

    return all_chunks
