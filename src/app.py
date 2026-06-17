import os
import sys
import shutil
import tempfile

# Add project root to sys.path to allow importing from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src import config

from src.ingest import load_document, chunk_documents, get_embedding_model, create_vector_store
from src.chain import build_rag_chain

# Global cache for embedding model to avoid reloading it on every upload
EMBEDDING_MODEL_CACHE = None

def get_cached_embeddings():
    global EMBEDDING_MODEL_CACHE
    if EMBEDDING_MODEL_CACHE is None:
        EMBEDDING_MODEL_CACHE = get_embedding_model()
    return EMBEDDING_MODEL_CACHE

def handle_upload(file_obj, progress=gr.Progress()):
    """
    Handles file upload, processes it, creates the vector store,
    and returns status update and states.
    """
    if file_obj is None:
        return "No file uploaded.", None, None, ""

    try:
        progress(0, desc="Initializing setup...")
        # Resolve Groq API key
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            return "Error: GROQ_API_KEY is not set in your .env file or environment variables.", None, None, ""

        # Create temporary file path to prevent naming collisions
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(file_obj.name))
        shutil.copy(file_obj.name, temp_path)
        
        progress(0.2, desc="Loading FAQ document...")
        documents = load_document(temp_path)
        
        progress(0.4, desc="Splitting document into chunks...")
        chunks = chunk_documents(documents)
        
        if not chunks:
            return "Error: No text chunks found in document.", None, None, ""

        progress(0.6, desc="Initializing embedding model (this may take a minute first time)...")
        embedder = get_cached_embeddings()
        
        progress(0.8, desc="Creating FAISS vector index...")
        vector_store = create_vector_store(chunks, embedder)
        
        progress(0.9, desc="Building QA Retrieval Chain...")
        qa_chain = build_rag_chain(vector_store, api_key=groq_key)

        status_msg = (
            f"**Indexing Complete!**\n\n"
            f"- **Filename:** `{os.path.basename(file_obj.name)}`\n"
            f"- **Total chunks created:** {len(chunks)}\n"
            f"- **Vector DB:** FAISS (in-memory)\n"
            f"- **Status:** Ready to answer questions!"
        )
        
        # Cleanup temporary files (FAISS is in memory)
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception:
            pass

        return status_msg, vector_store, qa_chain, ""

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(error_details)
        return f"**Error during ingestion:** {str(e)}", None, None, ""

def chat_respond(user_message, history, qa_chain_state):
    """
    Handles conversation and queries the RAG chain.
    """
    if not user_message.strip():
        return "", history, "Please type a message.", ""

    if qa_chain_state is None:
        # User tried to query without uploading a document
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "Please upload and index a PDF or TXT FAQ document first."})
        return "", history, "Document not indexed.", ""

    try:
        # Run QA Chain
        response = qa_chain_state.invoke({
            "query": user_message,
            "chat_history": history
        })
        answer = response["result"]
        source_documents = response.get("source_documents", [])

        # Format sources citation
        sources_text = ""
        if not source_documents:
            sources_text = "_No source chunks met the minimum similarity score cutoff (0.35). Fallback response triggered._"
        else:
            sources_text = "### Retrieved Sources (Score >= 0.35)\n\n"
            for i, doc in enumerate(source_documents, 1):
                score = doc.metadata.get("similarity_score", 0.0)
                source_info = os.path.basename(doc.metadata.get("source", "FAQ"))
                page_info = f" | Page {doc.metadata.get('page') + 1}" if doc.metadata.get("page") is not None else ""
                sources_text += (
                    f"**Source {i}:** `{source_info}{page_info}` (Score: **{score:.3f}**)\n"
                    f"```text\n{doc.page_content.strip()}\n```\n\n"
                )

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})
        return "", history, "Ready", sources_text

    except Exception as e:
        error_msg = f"**Error:** {str(e)}"
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history, error_msg, ""

# Custom Premium styling for Gradio to force a white/black minimal theme
custom_css = """
:root, .dark {
    --body-background-fill: #ffffff !important;
    --body-text-color: #000000 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f9fafb !important;
    --border-color-primary: #e5e7eb !important;
    --border-color-secondary: #e5e7eb !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e5e7eb !important;
    --block-title-text-color: #000000 !important;
    --block-label-text-color: #374151 !important;
    
    /* Input fields */
    --input-background-fill: #ffffff !important;
    --input-border-color: #e5e7eb !important;
    --input-text-color: #000000 !important;
    --input-placeholder-color: #9ca3af !important;
    
    /* Buttons (Black & White style instead of Blue) */
    --button-primary-background-fill: #000000 !important;
    --button-primary-background-fill-hover: #1f2937 !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-border-color: #000000 !important;
    
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-background-fill-hover: #f3f4f6 !important;
    --button-secondary-text-color: #000000 !important;
    --button-secondary-border-color: #e5e7eb !important;
}

/* Fix text legibility in chatbot bubbles & code blocks on dark backgrounds */
.message.user, .message.user *, .user, .user * {
    color: #ffffff !important;
}
.message.assistant, .message.assistant *, .assistant, .assistant * {
    color: #000000 !important;
}

pre, code, pre *, code * {
    color: #e5e7eb !important;
}

/* Force block labels (like Chat History, Ask a question, Upload FAQ Document, etc.) to have white text on their dark grey backgrounds */
.block-label, .block-label *, legend, legend *, .label, .label *, [data-testid="block-label"], [data-testid="block-label"] * {
    color: #ffffff !important;
}

/* Force file preview text (name, size, list items) to be white against dark backgrounds */
.file-preview, .file-preview *, .file-name, .file-name *, .file-size, .file-size *, .file-item, .file-item * {
    color: #ffffff !important;
}

#header {
    text-align: center;
    margin-bottom: 20px;
    padding: 20px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    color: #000000;
}
#header h1 {
    font-size: 2.5em;
    font-weight: 800;
    margin: 0;
    color: #000000;
}
#header p {
    font-size: 1.1em;
    margin-top: 10px;
    color: #374151;
}
.status-box {
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    background-color: #ffffff;
    color: #000000;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="neutral", secondary_hue="neutral"),
    css=custom_css
) as demo:
    # Vector store state & QA chain state per session
    vector_store_state = gr.State(None)
    qa_chain_state = gr.State(None)

    with gr.Group(elem_id="header"):
        gr.Markdown(
            """
            # FAQ RAG Chatbot
            Strict, hallucination-free Q&A grounded only in your uploaded company FAQ document.
            """
        )

    with gr.Row():
        # Configuration Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### Configuration & Document Indexer")
            
            file_upload = gr.File(
                label="Upload FAQ Document",
                file_types=[".pdf", ".txt"],
                file_count="single"
            )
            
            index_btn = gr.Button("Index Document", variant="primary")
            
            gr.Markdown("### Indexing Status")
            status_output = gr.Markdown(
                "Awaiting document upload...\n\nPlease upload a PDF or TXT FAQ and click 'Index Document'.",
                elem_classes="status-box"
            )

        # Chat and Sources Main Window
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", height=450)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Ask a question about the FAQ",
                    placeholder="e.g. What is the return policy?",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                
            clear_btn = gr.ClearButton([chatbot, msg_input], value="Clear Chat")
            
            gr.Markdown("---")
            sources_output = gr.Markdown("### Retrieved Sources\nNo queries made yet.")

    # Event handlers
    index_btn.click(
        fn=handle_upload,
        inputs=[file_upload],
        outputs=[status_output, vector_store_state, qa_chain_state, sources_output],
        show_progress=True
    )
    
    # Send message on click or enter key
    submit_event = gr.on(
        triggers=[msg_input.submit, submit_btn.click],
        fn=chat_respond,
        inputs=[msg_input, chatbot, qa_chain_state],
        outputs=[msg_input, chatbot, status_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=False
    )
