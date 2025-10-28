"""
Vector store management for emergency manual embeddings.

- Stores chunk embeddings in a lightweight SQLite DB (JSON vectors).
- Supports page-level indexing from markdown files under a manual directory.
- Records page_number and page_path in metadata for citation.
"""
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import (
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


class VectorStoreManager:
    """Manages vector embeddings storage and retrieval using sqlite-vec"""
    
    def __init__(self, db_path: Path | None = None):
        """Initialize vector store"""
        if db_path is None:
            db_path = VECTOR_DB_DIR / "manual_embeddings.db"
        
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database with vector support"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create table for document chunks and embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                section TEXT,
                page_number TEXT,
                metadata TEXT,
                embedding_json TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def chunk_document(self, text: str, section: str | None = None) -> List[Tuple[str, dict]]:
        """Split document into chunks with metadata"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Create chunks with metadata
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            # Try to extract section info from chunk
            section_id = self._extract_section_id(chunk) or section
            
            metadata = {
                'section': section_id,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            chunk_metadata.append((chunk, metadata))
        
        return chunk_metadata
    
    def _extract_section_id(self, text: str) -> str | None:
        """Extract section ID from text (e.g., G1, M2, S5, etc.)"""
        import re
        # Look for patterns like G1, M10, S15, etc.
        match = re.search(r'\b([GMSTUAR]\d{1,2})\b', text)
        if match:
            return match.group(1)
        return None
    
    def add_documents(self, text: str, section: str | None = None, page_number: str | None = None, page_path: str | None = None) -> None:
        """Add documents to vector store"""
        # Chunk the document
        chunks = self.chunk_document(text, section)
        
        # Generate embeddings for all chunks
        texts = [chunk[0] for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Store in database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for (content, metadata), embedding in zip(chunks, embeddings):
            # Enrich metadata with page info when present
            if page_number is not None:
                metadata['page_number'] = str(page_number)
            if page_path is not None:
                metadata['page_path'] = str(page_path)
            cursor.execute("""
                INSERT INTO document_chunks (content, section, page_number, metadata, embedding_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                content,
                metadata.get('section'),
                str(page_number) if page_number is not None else None,
                json.dumps(metadata, ensure_ascii=False),
                json.dumps(embedding)
            ))
        
        conn.commit()
        conn.close()
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for similar documents using cosine similarity"""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Retrieve all documents and compute similarity
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT content, section, page_number, metadata, embedding_json
            FROM document_chunks
            """
        )
        
        results = []
        for row in cursor.fetchall():
            content, section, page_number, metadata_json, embedding_json = row
            doc_embedding = json.loads(embedding_json)
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            metadata = json.loads(metadata_json) if metadata_json else {}
            if section and not metadata.get('section'):
                metadata['section'] = section
            if page_number and not metadata.get('page_number'):
                metadata['page_number'] = page_number
            results.append((content, metadata, similarity))
        
        conn.close()
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def clear_store(self):
        """Clear all documents from vector store"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_chunks")
        conn.commit()
        conn.close()
        print("Vector store cleared")
    
    def get_document_count(self) -> int:
        """Get total number of chunks in store"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # --- New: Manual directory indexing & helpers ---
    def rebuild_from_directory(self, manual_dir: Path) -> int:
        """Clear store and index markdown pages under `manual_dir`.

        Returns the total number of chunks indexed.
        """
        manual_dir = Path(manual_dir)
        if not manual_dir.exists():
            raise FileNotFoundError(f"Manual directory not found: {manual_dir}")

        self.clear_store()

        md_files = sorted([p for p in manual_dir.glob("*.md") if p.is_file()])
        total_chunks = 0

        for md_path in md_files:
            page_number = self._infer_page_number(md_path)
            text = md_path.read_text(encoding="utf-8", errors="ignore")
            before = self.get_document_count()
            self.add_documents(
                text,
                section=None,
                page_number=str(page_number) if page_number is not None else None,
                page_path=str(md_path),
            )
            after = self.get_document_count()
            total_chunks += max(0, after - before)

        return total_chunks

    def _infer_page_number(self, md_path: Path) -> int | None:
        """Infer page number from filename like '001.md', 'p_12.md', 'page-5.md'."""
        import re
        name = md_path.stem
        match = re.search(r"(\d{1,4})", name)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
        return None

    def unique_pages_from_results(
        self, results: Iterable[Tuple[str, Dict[str, Any], float]]
    ) -> List[Dict[str, Any]]:
        """Return unique pages sorted by best similarity.

        Each item: { 'page_number': int|str, 'score': float, 'page_path': str|None }
        """
        best: Dict[str, Dict[str, Any]] = {}
        for _content, metadata, sim in results:
            page_number = str(metadata.get("page_number")) if metadata.get("page_number") is not None else None
            page_path = metadata.get("page_path")
            if not page_number:
                continue
            current = best.get(page_number)
            if current is None or sim > current["score"]:
                best[page_number] = {"page_number": page_number, "score": sim, "page_path": page_path}
        # Sort by score desc
        return sorted(best.values(), key=lambda d: d["score"], reverse=True)

    def read_page_markdown(self, manual_dir: Path, page_number: int | str) -> str:
        """Read full markdown content of a page by number.

        Looks for files with first integer occurrence matching page_number.
        """
        manual_dir = Path(manual_dir)
        page_number = str(page_number)
        candidates = sorted([p for p in manual_dir.glob("*.md") if p.is_file()])
        for p in candidates:
            if page_number in p.stem or page_number.zfill(3) in p.stem:
                try:
                    return p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
        # As a fallback, return empty string
        return ""
