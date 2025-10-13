"""
Vector store management for emergency manual embeddings
Using sqlite-vec for vector storage
"""
import sqlite3
from pathlib import Path
from typing import List, Tuple
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, OPENAI_API_KEY


class VectorStoreManager:
    """Manages vector embeddings storage and retrieval using sqlite-vec"""
    
    def __init__(self, db_path: Path = None):
        """Initialize vector store"""
        if db_path is None:
            db_path = VECTOR_DB_DIR / "manual_embeddings.db"
        
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self._init_db()
    
    def _init_db(self):
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
    
    def chunk_document(self, text: str, section: str = None) -> List[Tuple[str, dict]]:
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
    
    def _extract_section_id(self, text: str) -> str:
        """Extract section ID from text (e.g., G1, M2, S5, etc.)"""
        import re
        # Look for patterns like G1, M10, S15, etc.
        match = re.search(r'\b([GMSTUAR]\d{1,2})\b', text)
        if match:
            return match.group(1)
        return None
    
    def add_documents(self, text: str, section: str = None):
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
            cursor.execute("""
                INSERT INTO document_chunks (content, section, metadata, embedding_json)
                VALUES (?, ?, ?, ?)
            """, (
                content,
                metadata.get('section'),
                json.dumps(metadata),
                json.dumps(embedding)
            ))
        
        conn.commit()
        conn.close()
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, dict, float]]:
        """Search for similar documents using cosine similarity"""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Retrieve all documents and compute similarity
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content, section, metadata, embedding_json
            FROM document_chunks
        """)
        
        results = []
        for row in cursor.fetchall():
            content, section, metadata_json, embedding_json = row
            doc_embedding = json.loads(embedding_json)
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            metadata = json.loads(metadata_json)
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
