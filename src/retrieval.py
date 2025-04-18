import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        
    def retrieve_by_source_id(self, source_id: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document_path = None
        target_document = None
        for path in self.documents_dir.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    if doc.get("metainfo", {}).get("company_name") == source_id:
                        document_path = path
                        target_document = doc
                        break
            except Exception as e:
                _log.warning(f"Error reading or parsing document {path.name}: {e}")
                continue
                    
        if document_path is None or target_document is None:
            raise ValueError(f"No document found with source ID '{source_id}'.")
            
        sha1_name = target_document.get("metainfo", {}).get("sha1_name")
        if not sha1_name:
             raise ValueError(f"Document with source ID '{source_id}' is missing 'sha1_name' in metainfo.")

        bm25_path = self.bm25_db_dir / f"{sha1_name}.pkl"
        if not bm25_path.exists():
             raise ValueError(f"BM25 index not found for document with source ID '{source_id}' at {bm25_path}")

        try:
            with open(bm25_path, 'rb') as f:
                bm25_index = pickle.load(f)
        except Exception as e:
            raise IOError(f"Error loading BM25 index for source ID '{source_id}': {e}")
            
        chunks = target_document.get("content", {}).get("chunks", [])
        pages = target_document.get("content", {}).get("pages", [])
        if not chunks or not pages:
             _log.warning(f"Document with source ID '{source_id}' has missing chunks or pages.")
             return []
        
        tokenized_query = query.split()
        try:
            scores = bm25_index.get_scores(tokenized_query)
        except Exception as e:
            _log.error(f"Error getting BM25 scores for source ID '{source_id}': {e}")
            return []
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next((page for page in pages if page.get("page") == chunk.get("page")), None)
            
            if parent_page is None:
                _log.warning(f"Could not find parent page for chunk {index} (page {chunk.get('page')}) in source ID '{source_id}'. Skipping chunk.")
                continue

            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page.get("text", "")
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk.get("text", "")
                }
                retrieval_results.append(result)
        
        return retrieval_results


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=os.getenv("GEMINI_BASE_URL"),
            timeout=None,
            max_retries=2
            )
        return llm
    
    @staticmethod
    def set_up_llm():
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=os.getenv("GEMINI_BASE_URL"),
            timeout=None,
            max_retries=2
            )
        return llm

    def _load_dbs(self):
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}
        
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model=os.getenv("GEMINI_EMBEDDING_MODEL"))
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_source_id(self, source_id: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.warning(f"Database entry for '{report.get('name')}' is missing 'metainfo'. Skipping.")
                continue
            
            # Check for various field names that might contain the document identifier
            # This ensures compatibility with different CSV formats (source_id, company_name, document_name)
            doc_id = metainfo.get("source_id", 
                      metainfo.get("company_name", 
                        metainfo.get("document_name")))
            
            if doc_id == source_id:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No database entry found matching source ID '{source_id}'.")
            raise ValueError(f"No document database found matching source ID '{source_id}'.")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document.get("content", {}).get("chunks", [])
        pages = document.get("content", {}).get("pages", [])

        if not chunks or not pages:
             _log.warning(f"Document with source ID '{source_id}' has missing chunks or pages.")
             return []
        
        actual_top_n = min(top_n, len(chunks))
        if actual_top_n <= 0:
             _log.warning(f"No chunks available to search for source ID '{source_id}'.")
             return []
        
        try:
            embedding = self.llm.embeddings.create(
                input=query,
                model=os.getenv("GEMINI_EMBEDDING_MODEL"),
            )
            embedding = embedding.data[0].embedding
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
        except Exception as e:
            _log.error(f"Error during embedding creation or vector search for source ID '{source_id}': {e}")
            raise IOError(f"Vector search failed for source ID '{source_id}': {e}")
    
        retrieval_results = []
        seen_pages = set()
        
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(chunks)]
        if len(valid_indices) != len(indices[0]):
             _log.warning(f"Vector search returned invalid indices for source ID '{source_id}'.")

        for i, index in enumerate(valid_indices):
            distance = distances[0][i]
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next((page for page in pages if page.get("page") == chunk.get("page")), None)
            if parent_page is None:
                 _log.warning(f"Could not find parent page for chunk {index} (page {chunk.get('page')}) in source ID '{source_id}'. Skipping chunk.")
                 continue

            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page.get("text", "")
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk.get("text", "")
                }
                retrieval_results.append(result)
            
        return retrieval_results

    def retrieve_all(self, source_id: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            
            # Check for various field names that might contain the document identifier
            doc_id = metainfo.get("source_id", 
                      metainfo.get("company_name", 
                        metainfo.get("document_name")))
            
            if doc_id == source_id:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No database entry found matching source ID '{source_id}'.")
            raise ValueError(f"No document database found matching source ID '{source_id}'.")
        
        document = target_report["document"]
        pages = document.get("content", {}).get("pages", [])
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p.get("page", 0)):
            result = {
                "distance": 0.5,
                "page": page.get("page"),
                "text": page.get("text", "")
            }
            if result["page"] is not None:
                 all_pages.append(result)
            
        return all_pages


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()
        
    def retrieve_by_source_id(
        self, 
        source_id: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            source_id: Unique identifier for the document source to search within
            query: Search query
            llm_reranking_sample_size: Number of initial results to retrieve from vector DB
            documents_batch_size: Number of documents to analyze in one LLM prompt
            top_n: Number of final results to return after reranking
            llm_weight: Weight given to LLM scores (0-1)
            return_parent_pages: Whether to return full pages instead of chunks
            
        Returns:
            List of reranked document dictionaries with scores
        """
        vector_results = self.vector_retriever.retrieve_by_source_id(
            source_id=source_id,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        
        return reranked_results[:top_n]
