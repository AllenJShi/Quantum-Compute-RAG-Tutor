# app.py
import streamlit as st
import requests
import time
import pandas as pd
from pathlib import Path

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/ask"  # Adjust if your API is hosted elsewhere

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
            if "company_name" in df.columns:  # Using the original column name for compatibility
                documents = df["company_name"].unique().tolist()
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
    question = st.text_input("Enter your question:", key="question_input")
    
    # --- Button and API Call ---
    if st.button("Get Answer", key="submit_button"):
        if question:
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
    
                    # --- Display Results ---
                    st.markdown("---")
                    st.subheader("Answer:")
                    
                    # Format the answer based on its type
                    answer = result.get("answer", "No answer provided.")
                    if isinstance(answer, list):
                        st.markdown("- " + "\n- ".join([str(item) for item in answer]))
                    elif answer is True:
                        st.markdown("**Yes**")
                    elif answer is False:
                        st.markdown("**No**")
                    else:
                        st.markdown(str(answer))
    
                    with st.expander("Show Reasoning Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Reasoning Summary:**")
                            st.caption(result.get("reasoning_summary", "N/A"))
                        with col2:
                            st.markdown("**Relevant Pages:**")
                            pages = result.get("relevant_pages", [])
                            if pages:
                                st.caption(", ".join(str(p) for p in pages))
                            else:
                                st.caption("N/A")
                        
                        st.markdown("**Step-by-Step Analysis:**")
                        analysis = result.get("step_by_step_analysis", "")
                        if analysis:
                            # Format numbered steps nicely
                            formatted_analysis = analysis.replace("\n1.", "\n\n1.")
                            for i in range(2, 10):
                                formatted_analysis = formatted_analysis.replace(f"\n{i}.", f"\n\n{i}.")
                            st.markdown(formatted_analysis)
                        else:
                            st.caption("N/A")
                        
                        st.caption(f"_Response time: {end_time - start_time:.2f} seconds_")
    
                except requests.exceptions.Timeout:
                     st.error("The request timed out. The backend might be taking too long.")
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to the backend API at {API_URL}. Is it running?")
                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")
                    # Try to show detail from API error response if available
                    try:
                        error_detail = e.response.json().get("detail", "No details provided.")
                        st.error(f"Backend Error: {error_detail}")
                    except Exception:
                        pass  # Ignore if parsing error detail fails
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter a question.")
    
    # --- Example Questions ---
    with st.expander("Example Questions"):
        st.markdown("""
        Try asking questions like:
        - "What is superposition in quantum computing?"
        - "Explain the difference between quantum bits and classical bits."
        - "What is the significance of entanglement in quantum computing?"
        """)
