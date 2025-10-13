"""
Q&A Bot for emergency manual using RAG with LangChain and GPT-4
"""
from typing import Dict, Any
from pathlib import Path
import json
import hashlib
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from .embeddings import VectorStoreManager
from config import OPENAI_API_KEY, OPENAI_MODEL, EMERGENCY_MANUAL_PATH


class ManualQABot:
    """Q&A Bot for emergency rescue procedures manual"""
    
    def __init__(self):
        """Initialize the Q&A bot"""
        self.vector_store = VectorStoreManager()
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            # temperature=0.1, # GPT-5 does not allow temperature configuration due to thinking models behavior
            openai_api_key=OPENAI_API_KEY
        )
        self.chat_history = []
        self._init_prompt()
    
    def _init_prompt(self):
        """Initialize the prompt template"""
        self.qa_prompt = PromptTemplate(
            template="""你是一個專業的緊急救護程序助手。你的任務是根據「新北市政府消防局緊急傷病患作業程序」手冊來回答問題。

參考資料：
{context}

對話歷史：
{chat_history}

使用者問題：{question}

請根據以上資料回答問題。回答時請：
1. 直接、清楚地回答問題
2. 引用具體的章節編號（如 G1、M2、S5 等）
3. 如果資料中有相關的流程或步驟，請列出重點
4. 如果無法從資料中找到答案，請明確說明
5. 使用專業但易懂的語言

回答：""",
            input_variables=["context", "chat_history", "question"]
        )

    def load_manual(self, manual_path: Path = EMERGENCY_MANUAL_PATH):
        """Load and index the emergency manual"""
        if not manual_path.exists():
            raise FileNotFoundError(f"Manual file not found: {manual_path}")

        print(f"Loading manual from: {manual_path}")

        # Read manual content
        with open(manual_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine whether we need to rebuild embeddings
        manual_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        meta_path = self.vector_store.db_path.with_suffix(".meta.json")
        existing_chunks = self.vector_store.get_document_count()

        if existing_chunks > 0 and meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as meta_file:
                    meta = json.load(meta_file)
                if meta.get("manual_hash") == manual_hash:
                    print(
                        f"Manual already indexed with {existing_chunks} chunks. "
                        "Skipping re-embedding."
                    )
                    return
            except json.JSONDecodeError:
                # Proceed with rebuild if metadata file is corrupt
                pass

        # Clear existing documents before rebuilding
        self.vector_store.clear_store()

        # Add to vector store
        self.vector_store.add_documents(content)

        chunk_count = self.vector_store.get_document_count()
        print(f"Manual loaded successfully. Total chunks: {chunk_count}")

        # Persist metadata for future runs
        meta_payload = {
            "manual_path": str(manual_path.resolve()),
            "manual_hash": manual_hash,
            "indexed_at": datetime.utcnow().isoformat()
        }
        with meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(meta_payload, meta_file, ensure_ascii=False, indent=2)
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the emergency manual
        
        Returns:
            dict with 'answer', 'sources', and 'chat_history'
        """
        # Search for relevant documents
        relevant_docs = self.vector_store.similarity_search(question, k=4)
        
        # Prepare context from relevant documents
        context_parts = []
        sources = []
        
        for i, (content, metadata, similarity) in enumerate(relevant_docs):
            section = metadata.get('section', '未知章節')
            context_parts.append(f"[章節 {section}]\n{content}")
            sources.append({
                'section': section,
                'content': content[:200] + "..." if len(content) > 200 else content,
                'similarity': similarity
            })
        
        context = "\n\n".join(context_parts)
        
        # Prepare chat history
        chat_history_text = ""
        for i, (q, a) in enumerate(self.chat_history[-3:]):  # Last 3 exchanges
            chat_history_text += f"問題 {i+1}: {q}\n回答 {i+1}: {a}\n\n"
        
        # Generate answer using LLM
        prompt = self.qa_prompt.format(
            context=context,
            chat_history=chat_history_text,
            question=question
        )
        
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Update chat history
        self.chat_history.append((question, answer))
        
        return {
            'answer': answer,
            'sources': sources,
            'chat_history': self.chat_history
        }
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
    
    def get_section_info(self, section_id: str) -> str:
        """Get information about a specific section"""
        # Search for the section
        relevant_docs = self.vector_store.similarity_search(section_id, k=2)
        
        if not relevant_docs:
            return f"找不到關於 {section_id} 的資訊"
        
        content = relevant_docs[0][0]
        return content
