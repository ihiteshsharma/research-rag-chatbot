import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizers warning

import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from backend.pdf_parser import extract_text_from_pdf, extract_tables_from_pdf, extract_images_from_pdf, process_pdf
from backend.preprocessing import clean_text, chunk_text
from backend.embeddings import generate_embeddings, create_faiss_index, retrieve_relevant_chunks, load_embeddings, save_embeddings
from backend.generation import generate_with_ollama, generate_with_claude
from backend.visualization import process_tables, extract_data_for_visualization, plot_data
from backend.utils import ensure_folder_exists

# Initialize session state
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit app
st.title("Academic Research Chatbot")

# Upload PDFs
uploaded_files = st.file_uploader("Upload research papers (PDFs)", type="pdf", accept_multiple_files=True)
if uploaded_files and st.session_state.knowledge_base is None:
    ensure_folder_exists("data/research_papers")
    all_text = ""
    all_tables = []
   
     # Save uploaded files
    for uploaded_file in uploaded_files:
        pdf_path = f"data/research_papers/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Process PDFs in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_pdf, [f"data/research_papers/{uploaded_file.name}" for uploaded_file in uploaded_files]))

    # Combine results
    for text, tables in results:
        all_text += text
        all_tables.extend(tables)

    # Preprocess text
    cleaned_text = clean_text(all_text)
    chunks = chunk_text(cleaned_text)

    # Generate embeddings and create FAISS index
    embeddings_file = "models/embeddings.npy"
    if os.path.exists(embeddings_file):
        embeddings = load_embeddings(embeddings_file)
    else:
        embeddings = generate_embeddings(chunks)
        save_embeddings(embeddings, embeddings_file)
        
    index = create_faiss_index(embeddings)
    st.session_state.knowledge_base = {"index": index, "chunks": chunks, "tables": all_tables}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve relevant chunks
    if st.session_state.knowledge_base:
        index = st.session_state.knowledge_base["index"]
        chunks = st.session_state.knowledge_base["chunks"]
        tables = st.session_state.knowledge_base["tables"]

        relevant_chunks = retrieve_relevant_chunks(query, index, chunks)

        # Prepare messages for Claude
        context = "\n".join(relevant_chunks)
        messages = [
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        # Include conversation history
        for msg in st.session_state.messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Generate response with Claude
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = generate_with_claude(messages)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Generate visualization if applicable
        if "plot" in query.lower():
            processed_tables = process_tables(tables)
            x, y = extract_data_for_visualization(processed_tables)
            st.pyplot(plot_data(x, y))
            