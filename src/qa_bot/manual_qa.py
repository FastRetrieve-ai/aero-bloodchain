"""
Q&A Bot for emergency manual using RAG (page-aware) with LangChain.

Key behaviors:
- Index each manual page (markdown) under data/emergency-patient-rescue-process
- Store page_number in embeddings metadata for citation
- Retrieve top-20 chunks, expand to their unique pages, and feed full page text to LLM
- Return answers with standardized citations (page numbers)
"""
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import hashlib
import re
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from .embeddings import VectorStoreManager
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    EMERGENCY_MANUAL_DIR,
    EMERGENCY_MANUAL_PDF,
)


class ManualQABot:
    """Q&A Bot for emergency rescue procedures manual (page-aware RAG)."""

    def __init__(self):
        """Initialize the Q&A bot."""
        self.vector_store = VectorStoreManager()
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            # temperature not configurable for some thinking models
            openai_api_key=OPENAI_API_KEY,
        )
        self.chat_history: list[tuple[str, str]] = []
        self.manual_dir: Path = Path(EMERGENCY_MANUAL_DIR)
        self._init_prompt()

    def _init_prompt(self) -> None:
        """Initialize the prompt template (grounded, page-citation)."""
        today = datetime.now().strftime('%Y-%m-%d')
        self.qa_prompt = PromptTemplate(
            template=(
                f"今天日期：{today}\n"
                "你是一位緊急救護程序的專業助手。請只根據下方提供的手冊內容來回答，"
                "不要臆測或加入未提供的資訊。若找不到答案，請明確說明。\n\n"
                "# 參考資料（多頁原文）：\n{context}\n\n"
                "# 對話歷史（摘要）：\n{chat_history}\n\n"
                "# 使用者問題：\n{question}\n\n"
                "請以繁體中文回覆，要求：\n"
                "1) 優先給出直接答案與要點條列；\n"
                "2) 嚴格根據參考資料原文，不可猜測；\n"
                "3) 需在回答中以 [p.頁碼] 形式點出關鍵出處，若有多頁，請以逗號分隔，例如 [p.001,p.003]；\n"
                "4) 若資料不足請說：『無法從手冊內容確定答案』。\n"
                "回答："
            ),
            input_variables=["context", "chat_history", "question"],
        )

    # --- Indexing: directory-based with change detection ---
    def build_or_load_index(self) -> int:
        """Build index from manual directory if changes detected.

        Returns the total chunk count in the vector store.
        """
        if not self.manual_dir.exists():
            raise FileNotFoundError(f"Manual directory not found: {self.manual_dir}")

        meta_path = self.vector_store.db_path.with_suffix(".meta.json")
        # Build a combined fingerprint of all markdown pages
        md_files = sorted([p for p in self.manual_dir.glob("*.md") if p.is_file()])
        hasher = hashlib.sha256()
        for p in md_files:
            try:
                hasher.update(p.name.encode("utf-8"))
                hasher.update(str(int(p.stat().st_mtime)).encode("utf-8"))
            except Exception:
                # Fall back to content when stat fails
                try:
                    hasher.update(p.read_bytes())
                except Exception:
                    pass
        manual_hash = hasher.hexdigest()

        existing_chunks = self.vector_store.get_document_count()

        if existing_chunks > 0 and meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                if (
                    meta.get("manual_dir") == str(self.manual_dir.resolve())
                    and meta.get("manual_hash") == manual_hash
                ):
                    return existing_chunks
            except Exception:
                pass

        # Rebuild index
        total_chunks = self.vector_store.rebuild_from_directory(self.manual_dir)
        meta_payload = {
            "manual_dir": str(self.manual_dir.resolve()),
            "manual_hash": manual_hash,
            "indexed_at": datetime.utcnow().isoformat(),
            "chunks": total_chunks,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)
        return total_chunks

    # --- Q&A ---
    def _chat_history_text(self, max_rounds: int = 3) -> str:
        text = ""
        for i, (q, a) in enumerate(self.chat_history[-max_rounds:]):
            text += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"
        return text

    def _normalize_page_number(self, value: Any) -> str | None:
        """Normalize page identifiers to digit-only strings (e.g., '005' -> '5')."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"(\d+)", text)
        if not match:
            return None
        digits = match.group(1)
        try:
            return str(int(digits))
        except Exception:
            stripped = digits.lstrip("0")
            return stripped or "0"

    def _compose_context_from_pages(
        self, pages: List[Dict[str, Any]], max_pages: int | None = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Read full page markdown for the given pages and build context block.

        Returns (context_text, citations)
        citations: list of {page_number, score, page_path}
        """
        selected = pages if max_pages is None else pages[:max_pages]
        parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        for p in selected:
            normalized_page = self._normalize_page_number(p.get("page_number"))
            if normalized_page is None:
                continue
            score = p.get("score")
            path = p.get("page_path")
            page_md = self.vector_store.read_page_markdown(
                self.manual_dir, normalized_page
            )
            header = f"[p.{str(normalized_page).zfill(3)}] 原文：\n"
            parts.append(header + page_md.strip())
            citations.append(
                {
                    "page_number": normalized_page,
                    "score": score,
                    "page_path": path,
                }
            )
        return "\n\n".join(parts), citations

    def _filter_citations_by_answer(
        self, answer_text: str, citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep only citations with pages explicitly mentioned in the answer.

        Supports patterns:
        - [p.005]
        - (p.005) or （p.005）
        - plain 'p.005' tokens
        """
        if not answer_text:
            return citations

        patterns = [
            re.compile(r"\[p\.\s*(\d+)\]", re.IGNORECASE),
            re.compile(r"[\(（]\s*p\.\s*(\d+)\s*[\)）]", re.IGNORECASE),
            re.compile(r"\bp\.\s*(\d{1,4})\b", re.IGNORECASE),
        ]

        mentioned_pages_set: set[str] = set()
        for pat in patterns:
            for match in pat.finditer(answer_text):
                page_norm = self._normalize_page_number(match.group(1))
                if page_norm:
                    mentioned_pages_set.add(page_norm)

        if not mentioned_pages_set:
            return citations

        # Preserve answer order when possible
        mentioned_pages: List[str] = []
        for pat in patterns:
            for match in pat.finditer(answer_text):
                page_norm = self._normalize_page_number(match.group(1))
                if page_norm and page_norm in mentioned_pages_set and page_norm not in mentioned_pages:
                    mentioned_pages.append(page_norm)

        filtered: List[Dict[str, Any]] = []
        for page in mentioned_pages:
            for citation in citations:
                citation_page = self._normalize_page_number(citation.get("page_number"))
                if citation_page == page and citation not in filtered:
                    filtered.append({**citation, "page_number": citation_page})
                    break

        return filtered if filtered else citations

    def ask(self, question: str, *, k_chunks: int = 20, max_pages: int | None = None) -> Dict[str, Any]:
        """Ask a question about the emergency manual.

        - Retrieves top-k_chunks similar chunks
        - Expands to unique pages and feeds full page content to the LLM
        - Returns an answer with standardized page citations
        """
        # Retrieve relevant chunks
        results = self.vector_store.similarity_search(question, k=k_chunks)
        unique_pages = self.vector_store.unique_pages_from_results(results)
        context, citations = self._compose_context_from_pages(unique_pages, max_pages=max_pages)

        # Prepare chat history text
        chat_history_text = self._chat_history_text()

        # Generate answer using LLM
        prompt = self.qa_prompt.format(
            context=context,
            chat_history=chat_history_text,
            question=question,
        )
        response = self.llm.invoke(prompt)
        answer_text = getattr(response, "content", "").strip()

        # Keep only citations explicitly referenced in the answer
        citations = self._filter_citations_by_answer(answer_text, citations)

        # try:
        #     sorted_citations = sorted(
        #         citations, key=lambda c: int(str(c.get("page_number") or 0))
        #     )
        # except Exception:
        #     sorted_citations = citations

        # citation_str_parts: List[str] = []
        # for c in sorted_citations:
        #     page_norm = self._normalize_page_number(c.get("page_number"))
        #     if page_norm is None:
        #         continue
        #     citation_str_parts.append(f"p.{page_norm.zfill(3)}")

        # if citation_str_parts:
        #     citation_str = ", ".join(citation_str_parts)
        #     answer_text = (
        #         answer_text
        #         + "\n\n—\n來源：新北市政府消防局《緊急傷病患作業程序》\n"
        #         + f"引用頁碼：{citation_str}"
        #     )
        #     if EMERGENCY_MANUAL_PDF.exists():
        #         answer_text += f"\n完整PDF：{EMERGENCY_MANUAL_PDF}"

        # Update chat history
        self.chat_history.append((question, answer_text))

        return {
            "answer": answer_text,
            "citations": citations,
            "chat_history": self.chat_history,
        }

    def clear_history(self) -> None:
        """Clear chat history"""
        self.chat_history = []

    def get_section_info(self, section_id: str) -> str:
        """Get raw content best-matching a section ID or code."""
        relevant_docs = self.vector_store.similarity_search(section_id, k=2)
        if not relevant_docs:
            return f"找不到關於 {section_id} 的資訊"
        return relevant_docs[0][0]
