# api.py
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os
import sys
import traceback  # Import traceback for detailed error logging

# Add src directory to sys.path to allow imports
# Use absolute path for robustness
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    # Ensure imports happen after sys.path modification
    from src.pipeline import Pipeline, configs, RunConfig, PipelineConfig
    from src.questions_processing import QuestionsProcessor
    from src.retrieval import (
        VectorRetriever,
        HybridRetriever,
    )  # Needed for type hinting if modifying processor
    from src.api_requests import (
        APIProcessor,
    )  # Needed for type hinting if modifying processor
    import src.prompts as prompts  # Import prompts to access schemas
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Attempted to add {src_path} to sys.path")
    print(f"Current sys.path: {sys.path}")
    print("Ensure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)

app = FastAPI(title="RAG Chatbot API")

# --- Configuration ---
DEFAULT_DATASET = "lecture"  # Changed default to lecture dataset
DEFAULT_CONFIG = "gemini_flash"  # Using the best config from README
BASE_DATA_PATH = project_root / "data"  # Use absolute path


# --- Request/Response Models ---
class QuestionRequest(BaseModel):
    question: str
    dataset_name: str = DEFAULT_DATASET
    config_name: str = DEFAULT_CONFIG
    target_document: str | None = None  # Optional parameter to explicitly target a specific document


class AnswerResponse(BaseModel):
    answer: str | float | int | bool | list[str] | None = (
        None  # Allow different answer types
    )
    step_by_step_analysis: str | None = None
    reasoning_summary: str | None = None
    relevant_pages: list[int] | None = None
    page_sources: list[dict] | None = None  # List of page-document mappings
    error: str | None = None  # Add error field
    # Add model statistics
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


# --- Helper Function to Get Processor ---
processor_cache = {}


# Make get_processor async to allow await inside if needed later
# Though initialization itself might be blocking, keep sync for now
def get_processor(dataset_name: str, config_name: str) -> QuestionsProcessor:
    cache_key = (dataset_name, config_name)
    if cache_key in processor_cache:
        print(f"Using cached processor for {cache_key}")
        return processor_cache[cache_key]

    root_path = BASE_DATA_PATH / dataset_name
    db_path = root_path / "databases"
    docs_path = db_path / "chunked_reports"
    subset_csv_path = root_path / "subset.csv"

    if not root_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Dataset directory not found at {root_path}"
        )
    if not db_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Databases directory not found at {db_path}"
        )
    if not docs_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Documents (chunked reports) directory not found at {docs_path}",
        )
    if not subset_csv_path.exists():
        print(
            f"Warning: subset.csv not found at {subset_csv_path}. Company identification might fail."
        )
        # Allow proceeding without subset.csv for now, but company extraction will be limited

    run_config = configs.get(config_name)
    if not run_config:
        raise HTTPException(
            status_code=400, detail=f"Config '{config_name}' not found in pipeline.py"
        )

    print(
        f"Initializing processor for dataset='{dataset_name}', config='{config_name}'..."
    )
    try:
        # Ensure paths are passed correctly
        processor = QuestionsProcessor(
            vector_db_dir=db_path / "vector_dbs",
            documents_dir=docs_path,
            questions_file_path=None,  # API doesn't use pre-defined questions file
            subset_path=subset_csv_path if subset_csv_path.exists() else None,
            # --- Pass relevant params from RunConfig ---
            parent_document_retrieval=run_config.parent_document_retrieval,
            llm_reranking=run_config.llm_reranking,
            llm_reranking_sample_size=run_config.llm_reranking_sample_size,
            top_n_retrieval=run_config.top_n_retrieval,
            parallel_requests=1,  # Force sequential for API calls within a single request
            api_provider=run_config.api_provider,
            answering_model=run_config.answering_model,
            full_context=run_config.full_context,
            new_challenge_pipeline=True,  # Assuming new pipeline structure based on code
        )
        processor_cache[cache_key] = processor
        print("Processor initialized.")
        return processor
    except Exception as e:
        print(f"Error initializing QuestionsProcessor: {e}")
        traceback.print_exc()  # Print full traceback for debugging
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize RAG processor: {str(e)}"
        )


# --- API Endpoint ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Receives a question and returns the answer generated by the RAG pipeline.
    """
    try:
        processor = get_processor(request.dataset_name, request.config_name)

        # --- Schema Inference (Streamlined) ---
        schema_patterns = {
            "number": ["how much", "what is the value", "what was the total", "number of", "how many"],
            "name": ["who", "what is the name"],
            "boolean": ["did ", "is there", "was there", "does ", "are "],
            "names": ["what are the names", "list the", "what are the"],
            "comparative": ["compare", "which ", "between"]
        }
        
        question_lower = request.question.lower()
        schema_type = "generic"  # Default schema
        
        # Check each pattern group and assign schema if matched
        for schema, patterns in schema_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                schema_type = schema
                break
                
        print(f"Processing question: '{request.question}' with inferred schema: '{schema_type}'")

        # --- Run RAG processing in a separate thread ---
        try:
            # Use asyncio.to_thread to avoid blocking the event loop
            # Pass the optional target_document_name if provided in the request
            answer_data = await asyncio.to_thread(
                processor.process_question,
                question=request.question,
                schema=schema_type,
                target_document_name=request.target_document  # Use the correct parameter name
            )
        except ValueError as ve:
             # Handle specific errors like "No document name found" or "No relevant context"
             print(f"ValueError during processing: {ve}")
             return AnswerResponse(error=str(ve))
        except Exception as proc_err:
             # Catch broader processing errors
             print(f"Error during processor.process_question: {proc_err}")
             traceback.print_exc()
             return AnswerResponse(error=f"Processing failed: {str(proc_err)}")

        # --- Format Response ---
        if not answer_data:
            return AnswerResponse(error="Processor returned no data.")

        # Handle errors returned by the processor
        if "error" in answer_data and answer_data["error"]:
            error_message = answer_data["error"]
            return AnswerResponse(error=error_message)

        # Extract answer information with cleaner fallback pattern
        # Start with direct fields from answer_data
        final_answer = answer_data.get("final_answer", answer_data.get("value"))
        step_by_step = answer_data.get("step_by_step_analysis")
        reasoning = answer_data.get("reasoning_summary")
        relevant_pages = answer_data.get("relevant_pages", [])
        page_sources = answer_data.get("page_sources", [])
        
        # If page_sources is missing but we have references with source_document info, extract from there
        if not page_sources and "references" in answer_data:
            page_sources = []
            for ref in answer_data.get("references", []):
                if "page_index" in ref and "source_document" in ref:
                    page_sources.append({"page_index": ref["page_index"], "source_document": ref["source_document"]})
        
        # Get model statistics
        model_stats = getattr(processor, 'response_data', {}) if hasattr(processor, 'response_data') else {}
        
        return AnswerResponse(
            answer=final_answer,
            step_by_step_analysis=step_by_step,
            reasoning_summary=reasoning,
            relevant_pages=relevant_pages,
            page_sources=page_sources,
            model=model_stats.get('model'),
            input_tokens=model_stats.get('input_tokens'),
            output_tokens=model_stats.get('output_tokens')
        )

    except HTTPException as http_exc:
        # Re-raise exceptions related to request validation or setup
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during the request handling
        print(f"Unhandled error in /ask endpoint: {e}")
        traceback.print_exc()  # Log the full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Run the API ---
if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    # Ensure CWD is project root when running `python api.py`
    os.chdir(project_root)
    print(f"Current working directory: {os.getcwd()}")
    # Use reload=True for development, remove for production
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
