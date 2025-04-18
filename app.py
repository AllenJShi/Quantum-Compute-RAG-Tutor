# app.py
import streamlit as st
import requests
import time
import pandas as pd
from pathlib import Path
import base64
import os
from PyPDF2 import PdfReader
import io

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/ask"  # Adjust if your API is hosted elsewhere

# --- Helper Functions for PDF Preview ---
def get_pdf_path(dataset_name, document_name):
    """Get the path to a PDF file based on dataset and document name."""
    # Check in the pdf_reports directory
    pdf_path = Path(f"data/{dataset_name}/pdf_reports/{document_name}.pdf")
    
    if pdf_path.exists():
        return pdf_path
    
    # If the document name already includes .pdf extension
    if document_name.lower().endswith('.pdf'):
        pdf_path = Path(f"data/{dataset_name}/pdf_reports/{document_name}")
        if pdf_path.exists():
            return pdf_path
    
    # If document name is the full file name
    pdf_dir = Path(f"data/{dataset_name}/pdf_reports")
    if pdf_dir.exists():
        for file in pdf_dir.glob("*.pdf"):
            # Check if the file stem matches or if the full name matches
            if file.stem == document_name or file.name == document_name:
                return file
            # Handle case where document_name is stored without extension in metadata
            if document_name in file.name:
                return file
    
    return None

def extract_page_from_pdf(pdf_path, page_number):
    """Extract a specific page from a PDF and return as bytes."""
    try:
        # Open the PDF file
        pdf = PdfReader(pdf_path)
        total_pages = len(pdf.pages)
        
        # Adjust for 0-based indexing if needed
        # Some systems use 1-based page numbers, but PyPDF2 uses 0-based indexing
        zero_based_index = page_number - 1 if page_number > 0 else page_number
        
        # Make sure the page exists
        if zero_based_index < 0 or zero_based_index >= total_pages:
            st.warning(f"Page {page_number} does not exist in the PDF. The PDF has {total_pages} pages.")
            return None
        
        # Create a new PDF with just the requested page
        from PyPDF2 import PdfWriter
        output = PdfWriter()
        output.add_page(pdf.pages[zero_based_index])
        
        # Write the selected page to bytes buffer
        output_buffer = io.BytesIO()
        output.write(output_buffer)
        output_buffer.seek(0)
        
        return output_buffer
    except Exception as e:
        st.error(f"Error extracting page from PDF: {e}")
        return None

def display_pdf_page(pdf_bytes):
    """Display a PDF page in the Streamlit app."""
    if pdf_bytes:
        # Encode PDF to base64
        base64_pdf = base64.b64encode(pdf_bytes.read()).decode('utf-8')
        
        # Display using HTML iframe
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="800px" 
            type="application/pdf"
            frameborder="0"
        ></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error("Could not load PDF page.")

# --- Initialize Session State ---
# Initialize session state to store values across reruns
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.question = ""
    st.session_state.answer = None
    st.session_state.showing_pdf = False
    st.session_state.viewing_page = None
    st.session_state.viewing_document = None
    st.session_state.dataset = None
    st.session_state.page_sources = {}
    st.session_state.step_by_step = None
    st.session_state.reasoning_summary = None
    st.session_state.relevant_pages = []
    st.session_state.response_time = None

# Function to handle PDF page viewing
def view_pdf_page(page_number, document_name, dataset_name):
    st.session_state.showing_pdf = True
    st.session_state.viewing_page = page_number
    st.session_state.viewing_document = document_name
    st.session_state.dataset = dataset_name

# --- Streamlit App Layout ---
st.set_page_config(page_title="Document Q&A Bot", layout="wide")
st.title("📄 Document Q&A Bot")
st.caption("Ask questions about your processed documents.")

# --- Dataset Selection ---
# Get available datasets by scanning data directory
data_dir = Path("data")
available_datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and (d / "databases").exists()]

if not available_datasets:
    st.error("No processed datasets found. Please process documents first.")
else:
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset", 
        options=available_datasets,
        index=available_datasets.index("lecture") if "lecture" in available_datasets else 0
    )
    
    # --- Config Selection ---
    config_options = ["gemini_flash", "max_nst_o3m", "gemini_thinking"]
    selected_config = st.sidebar.selectbox(
        "Select Model Config", 
        options=config_options, 
        index=0  # Default to gemini_flash
    )
    
    # --- Document Selection (Optional) ---
    # Try to load subset.csv to get document names
    documents = []
    subset_path = data_dir / selected_dataset / "subset.csv"
    try:
        if subset_path.exists():
            df = pd.read_csv(subset_path)
            # Check for different column names for document identification
            doc_column = None
            for possible_column in ['company_name', 'source_id', 'document_name']:
                if possible_column in df.columns:
                    doc_column = possible_column
                    break
            
            if doc_column:
                documents = df[doc_column].unique().tolist()
                documents = sorted(documents)
    except Exception as e:
        st.sidebar.warning(f"Could not load document list: {e}")
    
    target_document = None
    if documents:
        search_mode = st.sidebar.radio(
            "Search Mode",
            options=["Search Across All Documents", "Target Specific Document"],
            index=0
        )
        
        if search_mode == "Target Specific Document":
            target_document = st.sidebar.selectbox(
                "Target Document", 
                options=[""] + documents,
                format_func=lambda x: x if x else "Select a document..."
            )
            if not target_document:  # If empty string selected
                target_document = None
            else:
                st.sidebar.success(f"Targeting document: {target_document}")
    
    st.sidebar.info(f"Using Dataset: `{selected_dataset}`\nConfig: `{selected_config}`\nSearch Mode: {'Targeted' if target_document else 'All Documents'}")
    
    # --- Question Input ---
    question = st.text_input("Enter your question:", key="question_input", value=st.session_state.question)
    
    # --- Button and API Call ---
    if st.button("Get Answer", key="submit_button") and question:
        st.session_state.question = question  # Store question in session state
        st.session_state.showing_pdf = False  # Reset PDF viewing state
        start_time = time.time()
        
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "question": question,
                    "dataset_name": selected_dataset,
                    "config_name": selected_config,
                    "target_document": target_document  # Will be None if not specified
                }
                response = requests.post(API_URL, json=payload, timeout=180)  # Increased timeout
                response.raise_for_status()
                
                result = response.json()
                end_time = time.time()
                
                # Store results in session state
                st.session_state.answer = result.get("answer")
                st.session_state.step_by_step = result.get("step_by_step_analysis")
                st.session_state.reasoning_summary = result.get("reasoning_summary")
                st.session_state.relevant_pages = result.get("relevant_pages", [])
                st.session_state.page_sources = result.get("page_sources", [])
                st.session_state.response_time = end_time - start_time
                st.session_state.dataset = selected_dataset
                
                # Build a document-to-pages mapping for better organization
                doc_to_pages_mapping = {}
                for source in st.session_state.page_sources:
                    # Handle both formats (backend may return "page" or "page_index")
                    page_key = source.get("page", source.get("page_index"))
                    doc = source.get("document", source.get("source_document", "unknown"))
                    if page_key is not None:
                        if doc not in doc_to_pages_mapping:
                            doc_to_pages_mapping[doc] = []
                        doc_to_pages_mapping[doc].append(page_key)
                
                # Also keep the page-to-doc mapping for backward compatibility
                page_to_doc_mapping = {}
                for source in st.session_state.page_sources:
                    page_key = source.get("page", source.get("page_index"))
                    doc = source.get("document", source.get("source_document", "unknown"))
                    if page_key is not None:
                        if page_key not in page_to_doc_mapping:
                            page_to_doc_mapping[page_key] = []
                        page_to_doc_mapping[page_key].append(doc)
                
                st.session_state.doc_to_pages_mapping = doc_to_pages_mapping
                st.session_state.page_to_doc_mapping = page_to_doc_mapping
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.answer = None
    
    # --- Display PDF if viewing a page ---
    if st.session_state.showing_pdf and st.session_state.viewing_page is not None:
        pdf_path = get_pdf_path(st.session_state.dataset, st.session_state.viewing_document)
        
        if pdf_path and pdf_path.exists():
            st.markdown(f"### Viewing Document: {st.session_state.viewing_document}")
            st.markdown(f"**Page {st.session_state.viewing_page}**")
            
            # Create a back button to return to answer
            if st.button("← Back to Answer"):
                st.session_state.showing_pdf = False
                st.rerun()
            
            # Display the PDF page
            pdf_bytes = extract_page_from_pdf(pdf_path, st.session_state.viewing_page)
            if pdf_bytes:
                display_pdf_page(pdf_bytes)
            else:
                st.error(f"Could not extract page {st.session_state.viewing_page} from document.")
        else:
            st.error(f"Could not find PDF file for document: {st.session_state.viewing_document}")
            if st.button("← Back to Answer"):
                st.session_state.showing_pdf = False
                st.rerun()
    
    # --- Display Answer (when not showing PDF) ---
    elif st.session_state.answer is not None:
        st.markdown("---")
        st.subheader("Answer:")
        
        # Format the answer based on its type
        answer = st.session_state.answer
        if isinstance(answer, list):
            st.markdown("- " + "\n- ".join([str(item) for item in answer]))
        elif answer is True:
            st.markdown("**Yes**")
        elif answer is False:
            st.markdown("**No**")
        else:
            st.markdown(str(answer))
        
        # Display referenced pages grouped by document
        if st.session_state.doc_to_pages_mapping:
            st.markdown("### Referenced Pages")
            st.markdown("Click on a page number to view the source content:")
            
            for doc, pages in st.session_state.doc_to_pages_mapping.items():
                with st.expander(f"Document: {doc}"):
                    cols = st.columns(min(5, len(pages)))
                    for i, page in enumerate(pages):
                        col_index = i % len(cols)
                        with cols[col_index]:
                            if st.button(f"Page {page}", key=f"page_{page}_{doc}"):
                                view_pdf_page(page, doc, st.session_state.dataset)
                                st.rerun()
        
        # Display reasoning details
        with st.expander("Show Reasoning Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Reasoning Summary:**")
                st.caption(st.session_state.reasoning_summary or "N/A")
            with col2:
                st.markdown("**Relevant Pages:**")
                pages = st.session_state.relevant_pages
                if pages:
                    st.caption(", ".join(str(p) for p in pages))
                else:
                    st.caption("N/A")
            
            st.markdown("**Step-by-Step Analysis:**")
            analysis = st.session_state.step_by_step
            if analysis:
                # Format numbered steps nicely
                formatted_analysis = analysis.replace("\n1.", "\n\n1.")
                for i in range(2, 10):
                    formatted_analysis = formatted_analysis.replace(f"\n{i}.", f"\n\n{i}.")
                st.markdown(formatted_analysis)
            else:
                st.caption("N/A")
            
            if st.session_state.response_time:
                st.caption(f"_Response time: {st.session_state.response_time:.2f} seconds_")
    
    # --- Example Questions ---
    with st.expander("Example Questions"):
        st.markdown("""
        Try asking questions like:
        - "What is superposition in quantum computing?"
        - "Explain the difference between quantum bits and classical bits."
        - "What is the significance of entanglement in quantum computing?"
        """)
