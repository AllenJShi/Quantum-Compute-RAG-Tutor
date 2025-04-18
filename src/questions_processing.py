import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string"""
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, document_name: str) -> list:
        # Load document metadata
        if self.subset_path is None:
            # Allow proceeding without subset if not strictly needed for references,
            # but references might be incomplete. Log a warning.
            print("Warning: subset_path not provided. Cannot extract SHA1 for references.")
            document_sha1 = ""  # Or handle differently if SHA1 is critical
        else:
            # Load the mapping (assuming subset.csv maps document_name to sha1)
            try:
                self.documents_df = pd.read_csv(self.subset_path)
                # Check if dataframe has 'source_id' column, if not, use 'company_name' for backward compatibility
                id_column = 'source_id' if 'source_id' in self.documents_df.columns else 'company_name'
                self.id_column = id_column  # Store for reuse
                # Find the document's SHA1 from the subset CSV
                matching_rows = self.documents_df[self.documents_df[id_column] == document_name]
                if matching_rows.empty:
                    print(f"Warning: Could not find SHA1 for document '{document_name}' in {self.subset_path}")
                    document_sha1 = ""
                else:
                    document_sha1 = matching_rows.iloc[0]['sha1']
            except FileNotFoundError:
                print(f"Warning: subset_path '{self.subset_path}' not found. Cannot extract SHA1 for references.")
                document_sha1 = ""
            except Exception as e:
                print(f"Warning: Error reading subset_path '{self.subset_path}': {e}. Cannot extract SHA1 for references.")
                document_sha1 = ""

        refs = []
        for page in pages_list:
            refs.append({"pdf_sha1": document_sha1, "page_index": page})
        return refs

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2, max_pages: int = 8) -> list:
        """
        Validate that all page numbers mentioned in the LLM's answer are actually from the retrieval results.
        If fewer than min_pages valid references remain, add top pages from retrieval results.
        """
        if claimed_pages is None:
            claimed_pages = []
        
        retrieved_pages = [result['page'] for result in retrieval_results]
        
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]
        
        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")
        
        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    
                    if len(validated_pages) >= min_pages:
                        break
        
        if len(validated_pages) > max_pages:
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]
        
        return validated_pages

    def get_answer_for_document(self, document_name: str, question: str, schema: str) -> dict:
        """Retrieves context and generates an answer for a specific document."""
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )

        # Use the document_name as the source_id
        if self.full_context:
            # Call the renamed method with the renamed parameter
            retrieval_results = retriever.retrieve_all(source_id=document_name)
        else:
            # Call the renamed method with the renamed parameter
            retrieval_results = retriever.retrieve_by_source_id(
                source_id=document_name,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )

        if not retrieval_results:
            raise ValueError(f"No relevant context found in document '{document_name}' for the question.")

        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data

        # Add validated pages and references (using document_name)
        pages = answer_dict.get("relevant_pages", [])
        validated_pages = self._validate_page_references(pages, retrieval_results)
        answer_dict["relevant_pages"] = validated_pages
        
        # Store page-document pairs as a list of objects instead of a dictionary
        # This allows multiple entries for the same page number from different documents
        page_sources = []
        for page in validated_pages:
            page_sources.append({
                "page": page,
                "document": document_name
            })
        answer_dict["page_sources"] = page_sources
        
        # Create references with source document information
        references = []
        for page in validated_pages:
            # Include both SHA1 (for compatibility) and source_document (for our UI)
            ref = {
                "page_index": page,
                "source_document": document_name
            }
            # Add SHA1 if available
            if self.subset_path and self.subset_path.exists():
                try:
                    if not hasattr(self, 'documents_df') or self.documents_df is None:
                        self.documents_df = pd.read_csv(self.subset_path)
                    id_column = getattr(self, 'id_column', 'company_name')
                    matching_rows = self.documents_df[self.documents_df[id_column] == document_name]
                    if not matching_rows.empty:
                        ref["pdf_sha1"] = matching_rows.iloc[0]['sha1']
                except Exception:
                    pass
            references.append(ref)
            
        answer_dict["references"] = references

        return answer_dict

    def _extract_document_names_from_subset(self, question_text: str) -> list[str]:
        """
        Extract document names from a question by matching against names in the subset file.
        Uses multiple strategies to find document references:
        1. Exact matches (with word boundaries)
        2. Partial matches for long document names
        3. Keyword-based matching for specific topics
        """
        if self.subset_path is None or not self.subset_path.exists():
            print("Warning: subset_path not available for document name extraction from question.")
            return []  # Cannot extract without the subset file

        # Ensure DataFrame is loaded only once or if needed
        if not hasattr(self, 'documents_df') or self.documents_df is None:
            try:
                self.documents_df = pd.read_csv(self.subset_path)
                
                id_column = 'source_id' if 'source_id' in self.documents_df.columns else 'document_name'
                self.id_column = id_column  # Store for reuse
                
            except Exception as e:
                print(f"Error loading subset file '{self.subset_path}': {e}")
                return []

        found_documents = []
        id_column = getattr(self, 'id_column', 'document_name')

        # Get all document names from the subset file
        document_names = sorted(self.documents_df[id_column].unique(), key=len, reverse=True)
        
        # Strategy 1: Direct regex matching with word boundaries
        temp_question_text = question_text.lower()  # Work on a lowercase copy
        for doc_name in document_names:
            # First try exact match with word boundaries
            escaped_doc_name = re.escape(doc_name)
            pattern = rf'\b{escaped_doc_name}\b'  # Match whole words only
            
            if re.search(pattern, temp_question_text, re.IGNORECASE):
                found_documents.append(doc_name)
                # Remove the found name to avoid re-matching parts of it
                temp_question_text = re.sub(pattern, '', temp_question_text, flags=re.IGNORECASE)
        
        # If no direct matches were found, try more flexible approaches
        if not found_documents:
            # Strategy 2: For very long document names (like lecture documents), check for significant substrings
            for doc_name in document_names:
                # For long names, try to match by the most distinctive parts
                if len(doc_name) > 30 and "Module" in doc_name:
                    module_match = re.search(r'Module-(\d+)', doc_name)
                    if module_match and f"Module {module_match.group(1)}" in question_text:
                        found_documents.append(doc_name)
                        continue
                
                # If document has shortened name patterns like "PS1" or "Problem Set 1"
                if doc_name.startswith("PS") and len(doc_name) <= 5:
                    ps_number = doc_name[2:]
                    if f"PS{ps_number}" in question_text or f"Problem Set {ps_number}" in question_text:
                        found_documents.append(doc_name)
                        continue
            
            # Strategy 3: Topic-based matching (if no direct document reference)
            # If we have lecture documents and the question mentions specific quantum topics
            quantum_topics = {
                "superposition": ["Module-1", "Module 1"],
                "qubit": ["Module-1", "Module 1"],
                "measurement": ["Module-2", "Module 2"],
                "entanglement": ["Module-3", "Module 3"],
                "quantum gate": ["Module-4", "Module 4"],
                "quantum algorithm": ["Module-5", "Module 5"],
                "quantum circuit": ["Module-4", "Module 4"]
            }
            
            # Check if question contains any of these topics
            for topic, related_modules in quantum_topics.items():
                if topic in question_text.lower():
                    # Find documents that match these modules
                    for doc_name in document_names:
                        for module_ref in related_modules:
                            if module_ref in doc_name:
                                if doc_name not in found_documents:
                                    found_documents.append(doc_name)
        
        # Remove duplicates while preserving order
        unique_docs = []
        for doc in found_documents:
            if doc not in unique_docs:
                unique_docs.append(doc)
                
        return unique_docs

    def process_question(self, question: str, schema: str, target_document_name: Optional[str] = None):
        """
        Processes a question, either for a specific target document or by extracting
        document names from the question text. If no document is specified or extracted,
        it searches across all available documents.
        """
        extracted_document_names = []

        if target_document_name:
            # If a specific document is targeted, use it directly
            print(f"Processing question for specified document: '{target_document_name}'")
            extracted_document_names = [target_document_name]
        else:
            # Otherwise, try to extract document names from the question
            print("No target document specified, attempting extraction from question...")
            extracted_document_names = self._extract_document_names_from_subset(question)
            if extracted_document_names:
                print(f"Extracted document names from question: {extracted_document_names}")

        if not extracted_document_names:
            # --- Search Across All Documents ---
            print("No specific document identified. Searching across all available documents...")
            all_doc_names = []
            
            # Use id_column attribute if set, otherwise default to 'company_name' for backward compatibility
            id_column = getattr(self, 'id_column', 'company_name')
            
            if hasattr(self, 'documents_df') and self.documents_df is not None:
                 all_doc_names = self.documents_df[id_column].unique().tolist()
            elif self.subset_path and self.subset_path.exists():
                 try:
                     self.documents_df = pd.read_csv(self.subset_path)
                     # Check if dataframe has 'source_id' column, if not, use 'company_name'
                     id_column = 'source_id' if 'source_id' in self.documents_df.columns else 'company_name'
                     self.id_column = id_column  # Store for reuse
                     all_doc_names = self.documents_df[id_column].unique().tolist()
                 except Exception as e:
                     print(f"Error loading subset file '{self.subset_path}' to get all document names: {e}")
            
            if not all_doc_names:
                 return {
                    "error": "No documents found in the dataset. Please ensure documents are properly processed and indexed.",
                    "value": "N/A" 
                 }

            print(f"Found {len(all_doc_names)} documents to search.")
            
            # Initialize retriever
            if self.llm_reranking:
                retriever = HybridRetriever(vector_db_dir=self.vector_db_dir, documents_dir=self.documents_dir)
            else:
                retriever = VectorRetriever(vector_db_dir=self.vector_db_dir, documents_dir=self.documents_dir)

            all_retrieval_results = []
            # Track results by document to ensure balanced retrieval
            doc_results = {}
            
            # Define max chunks to retrieve per document to ensure balance
            chunks_per_doc = max(2, min(5, self.top_n_retrieval // len(all_doc_names) if len(all_doc_names) > 0 else 1))
            total_chunks_target = min(self.top_n_retrieval * 2, len(all_doc_names) * chunks_per_doc)
            
            print(f"Retrieving up to {chunks_per_doc} chunks per document, targeting {total_chunks_target} total chunks")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
                future_to_doc = {
                    executor.submit(
                        retriever.retrieve_by_source_id,
                        source_id=doc_name,
                        query=question,
                        llm_reranking_sample_size=self.llm_reranking_sample_size,
                        top_n=chunks_per_doc * 2,  # Get more than needed to allow for filtering
                        return_parent_pages=self.return_parent_pages
                    ): doc_name for doc_name in all_doc_names
                }
                
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc_name = future_to_doc[future]
                    try:
                        results = future.result()
                        if results:
                            # Add source document info to each result
                            for result in results:
                                result['source_document'] = doc_name
                            
                            # Store results by document
                            doc_results[doc_name] = results
                    except Exception as exc:
                        print(f"Retrieval failed for document '{doc_name}': {exc}")
            
            # Gather top chunks from each document based on relevance
            if doc_results:
                # First, sort each document's results by distance (most relevant first)
                for doc_name, results in doc_results.items():
                    results.sort(key=lambda x: x.get('distance', float('inf')))
                    # Only keep the top chunks_per_doc results
                    doc_results[doc_name] = results[:chunks_per_doc]
                
                # Combine all results
                for results in doc_results.values():
                    all_retrieval_results.extend(results)
                
                # Sort all results by relevance
                all_retrieval_results.sort(key=lambda x: x.get('distance', float('inf')))
                
                # Limit to top_n_retrieval * 2 to prevent context length issues
                all_retrieval_results = all_retrieval_results[:total_chunks_target]
                
                print(f"Gathered {len(all_retrieval_results)} relevant chunks from {len(doc_results)} documents")
            
            if not all_retrieval_results:
                raise ValueError("No relevant context found across any documents for the question.")
            
            # Group results by source document for better context organization
            grouped_results = {}
            for result in all_retrieval_results:
                source = result.get('source_document', 'unknown')
                if source not in grouped_results:
                    grouped_results[source] = []
                grouped_results[source].append(result)
            
            # Format results in a more structured way, grouping by document
            context_parts = []
            for source, results in grouped_results.items():
                # Add document header
                context_parts.append(f"Information from document: {source}")
                
                # Add text from each chunk
                for result in results:
                    page_number = result['page']
                    text = result['text']
                    context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')
                
                # Add separator between documents
                context_parts.append("---")
            
            rag_context = "\n\n".join(context_parts)
            
            # Get answer with the improved context
            answer_dict = self.openai_processor.get_answer_from_rag_context(
                question=question,
                rag_context=rag_context,
                schema=schema,
                model=self.answering_model
            )
            self.response_data = self.openai_processor.response_data

            pages_mentioned = answer_dict.get("relevant_pages", [])
            
            # Create a mapping of page numbers to source documents
            page_to_doc_mapping = {}
            for result in all_retrieval_results:
                page_num = result['page']
                source_doc = result.get('source_document', 'unknown')
                page_to_doc_mapping[page_num] = source_doc
            
            # Validate pages and associate each with its source document
            validated_pages = []
            page_sources = []  # Using a list of objects instead of dict for duplicate page numbers
            
            for p in pages_mentioned:
                if p in page_to_doc_mapping:
                    validated_pages.append(p)
                    page_sources.append({
                        "page": p,
                        "document": page_to_doc_mapping[p]
                    })
            
            # If we don't have enough validated pages, add some from retrieval results
            if len(validated_pages) < 2 and all_retrieval_results:
                existing_pages = set(validated_pages)
                for result in all_retrieval_results:
                    page = result['page']
                    if page not in existing_pages:
                        validated_pages.append(page)
                        page_sources.append({
                            "page": page,
                            "document": result.get('source_document', 'unknown')
                        })
                        existing_pages.add(page)
                        if len(validated_pages) >= 2:
                            break
            
            # Add the validated pages and their sources to the answer
            answer_dict["relevant_pages"] = validated_pages
            answer_dict["page_sources"] = page_sources
            
            # Format references with source document information
            references = []
            for page in validated_pages:
                source_doc = next((ps["document"] for ps in page_sources if ps["page"] == page), 'unknown')
                references.append({
                    "page_index": page,
                    "source_document": source_doc
                })
            
            answer_dict["references"] = references

            return answer_dict
            # --- End Search Across All Documents ---

        elif len(extracted_document_names) == 1:
            # --- Process Single Document ---
            document_name = extracted_document_names[0]
            answer_dict = self.get_answer_for_document(document_name=document_name, question=question, schema=schema)
            return answer_dict
        else:
            # --- Process Comparative Question ---
            print(f"Processing comparative question for documents: {extracted_document_names}")
            return self.process_comparative_question(question, extracted_document_names, schema)

    def process_comparative_question(self, question: str, document_names: List[str], schema: str) -> dict:
        """
        Process a question involving multiple documents in parallel:
        1. Rephrase the comparative question into individual questions
        2. Process each individual question using parallel threads
        3. Combine results into final comparative answer
        """
        # Step 1: Rephrase the comparative question (using document_names)
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=document_names  # Pass document_names to the rephrasing prompt
        )

        individual_answers = {}
        aggregated_references = []

        # Step 2: Process each individual question in parallel
        def process_document_question(doc_name: str) -> tuple[str, dict]:
            """Helper function to process one document's question and return (doc_name, answer)"""
            sub_question = rephrased_questions.get(doc_name)
            if not sub_question:
                # Attempt to use original question if rephrasing failed for some reason
                print(f"Warning: Could not generate sub-question for document: {doc_name}. Using original question.")
                sub_question = question  # Fallback, might not be ideal

            sub_schema = "number" if schema == "comparative" else schema

            answer_dict = self.get_answer_for_document(
                document_name=doc_name,
                question=sub_question,
                schema=sub_schema
            )
            return doc_name, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_doc = {
                executor.submit(process_document_question, doc_name): doc_name
                for doc_name in document_names
            }

            for future in concurrent.futures.as_completed(future_to_doc):
                try:
                    doc_name, answer_dict = future.result()
                    individual_answers[doc_name] = answer_dict

                    doc_references = answer_dict.get("references", [])
                    aggregated_references.extend(doc_references)
                except Exception as e:
                    doc_name = future_to_doc[future]
                    print(f"Error processing document {doc_name}: {str(e)}")
                    raise

        # Remove duplicate references (based on SHA1 and page index)
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            if key[0] is not None and key[1] is not None:
                unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())

        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data

        comparative_answer["references"] = aggregated_references
        return comparative_answer
