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
    error: str | None = None  # Add error field


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

        # --- Schema Inference (Generic Approach) ---
        question_lower = request.question.lower()
        if any(kw in question_lower for kw in ["how much", "what is the value", "what was the total", "number of", "how many"]):
            schema_type = "number"
        elif any(kw in question_lower for kw in ["who", "what is the name"]):
             schema_type = "name"
        elif any(kw in question_lower for kw in ["did ", "is there", "was there", "does ", "are "]):
             schema_type = "boolean"
        elif any(kw in question_lower for kw in ["what are the names", "list the", "what are the"]):
             schema_type = "names"
        elif "compare" in question_lower or "which " in question_lower or "between" in question_lower:
             schema_type = "comparative"
        else:
             schema_type = "generic" # Use the new generic schema

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

        # Handle potential errors returned by the processor itself
        if "error" in answer_data and answer_data["error"]:
            print(f"Processor returned error: {answer_data['error']}")
            # Try to get details if available
            detail_ref = answer_data.get("answer_details", {}).get("$ref")
            error_detail_info = None
            if detail_ref and detail_ref.startswith("#/answer_details/"):
                try:
                    index = int(detail_ref.split("/")[-1])
                    if (
                        0 <= index < len(processor.answer_details)
                        and processor.answer_details[index]
                    ):
                        error_detail_info = processor.answer_details[index].get(
                            "error_traceback"
                        )
                except (ValueError, IndexError, AttributeError):
                    pass  # Ignore if details aren't accessible
            error_message = answer_data["error"]
            if error_detail_info:
                error_message += f" | Details: {error_detail_info[:500]}..."  # Truncate long tracebacks

            return AnswerResponse(error=error_message)

        # Extract results based on the new_challenge_pipeline structure
        final_answer = answer_data.get("value")  # 'value' used in new pipeline
        references = answer_data.get("references", [])

        # Extract details from the answer_details using the $ref
        step_by_step = None
        reasoning = None
        relevant_pages_from_detail = None
        detail_ref = answer_data.get("answer_details", {}).get("$ref")

        if detail_ref and detail_ref.startswith("#/answer_details/"):
            try:
                # The index is stored during processing, access it via processor instance
                index = int(detail_ref.split("/")[-1])
                if (
                    hasattr(processor, "answer_details")
                    and 0 <= index < len(processor.answer_details)
                    and processor.answer_details[index]
                ):
                    details = processor.answer_details[index]
                    step_by_step = details.get("step_by_step_analysis")
                    reasoning = details.get("reasoning_summary")
                    # Get relevant_pages from details, fallback to main answer_dict if needed
                    relevant_pages_from_detail = details.get("relevant_pages")
                else:
                    print(
                        f"Warning: Could not find answer details for ref {detail_ref}"
                    )
            except (ValueError, IndexError, AttributeError) as detail_err:
                print(
                    f"Warning: Error accessing answer details for ref {detail_ref}: {detail_err}"
                )
                # Fallback or ignore if details cannot be accessed

        # Consolidate relevant pages (prefer details, fallback to main dict's validated pages)
        # The main `answer_data` should already have validated pages if `new_challenge_pipeline` is True
        relevant_pages_final = relevant_pages_from_detail  # Prefer pages from detailed analysis if available

        # If not in details, try getting validated pages from the main answer dict
        if relevant_pages_final is None:
            relevant_pages_final = answer_data.get(
                "relevant_pages"
            )  # These should be validated ones

        # If still None, extract from references (these are 0-based, convert back to 1-based for consistency if needed)
        # Note: The processor already validates pages, so this might be redundant unless details are missing
        if relevant_pages_final is None and references:
            # References are dicts like {"pdf_sha1": ..., "page_index": 0}
            # Convert 0-based index back to 1-based for display consistency if needed
            # Assuming the processor's internal logic uses 1-based, let's stick to what it provides
            # relevant_pages_final = sorted(list(set(ref.get("page_index", -1) + 1 for ref in references if ref.get("page_index", -1) >= 0)))
            # Let's trust answer_data["relevant_pages"] as the primary source of 1-based pages
            pass  # Keep relevant_pages_final as None if not found elsewhere

        return AnswerResponse(
            answer=final_answer,
            step_by_step_analysis=step_by_step,
            reasoning_summary=reasoning,
            relevant_pages=relevant_pages_final,
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
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)
