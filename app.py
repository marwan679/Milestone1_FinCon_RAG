import os
import gradio as gr
from src.generation.chain import FinConRAGPipeline
from src.ingestion.run_ingestion import run_ingestion_pipeline

try:
    import spaces
    def gpu_decorator(fn):
        return spaces.GPU(fn)
except Exception:
    def gpu_decorator(fn):
        return fn

# Initialize Pipeline
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = FinConRAGPipeline()
    return pipeline

SAMPLE_QUERIES = [
    "What is the mandatory timeline for Form 8-K disclosure under SEC Item 1.05 for material cybersecurity incidents?",
    "What are the differences between Scope 1, Scope 2, and Scope 3 emissions under IFRS S2 in 2026?",
    "What are the eligibility criteria and turnover thresholds for EU CSRD Wave 2 compliance in 2026?",
    "What is the Basel III Endgame capital output floor percentage for risk-weighted assets?",
    "What confidence level is required for Expected Shortfall under the FRTB market risk framework?"
]

@gpu_decorator
def handle_ingestion():
    try:
        run_ingestion_pipeline()
        global pipeline
        if pipeline:
            pipeline.refresh_parent_store()
        return "✅ Document Ingestion & Hybrid Indexing completed successfully!"
    except Exception as e:
        return f"❌ Ingestion failed: {str(e)}"

@gpu_decorator
def handle_query(query: str, top_k: int):

    if not query or not query.strip():
        return (
            "⚠️ Please enter a valid financial question.",
            "",
            "",
            "",
            ""
        )
    
    pipe = get_pipeline()
    result = pipe.run(query=query.strip(), top_k_rerank=int(top_k))
    
    # 1. Answer markdown
    answer_md = result.get("answer", "No answer generated.")
    
    # 2. Query expansion & Entities
    entities = result.get("entities", [])
    ent_str = ", ".join([f"{e['text']} ({e['label']})" for e in entities]) if entities else "None"
    expanded = result.get("expanded_query", query)
    metadata_info = f"**Recognized Entities (spaCy):** {ent_str}\n\n**Expanded Query:** `{expanded}`"
    
    # 3. Sources
    sources = result.get("sources", [])
    sources_md = "### 📚 Source Attributions\n" + "\n".join([f"- 📄 `{src}`" for src in sources]) if sources else "No specific sources cited."
    
    # 4. Child chunks details
    chunks_md = ""
    for idx, chunk in enumerate(result.get("retrieved_child_chunks", []), start=1):
        chunks_md += f"#### Rank {idx}: `{chunk.get('id')}`\n"
        chunks_md += f"- **Rerank Score:** `{chunk.get('rerank_score', 0.0):.4f}` | **RRF Score:** `{chunk.get('rrf_score', 0.0):.4f}`\n"
        chunks_md += f"- **Parent ID:** `{chunk.get('parent_id')}`\n"
        chunks_md += f"```text\n{chunk.get('text', '')}\n```\n\n---\n"
    
    # 5. Parent context
    parent_context = result.get("context_used", "")
    
    return answer_md, metadata_info, sources_md, chunks_md, parent_context

# Custom CSS for styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
"""

with gr.Blocks(title="FinCon & Regulatory RAG Assistant", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💼 FinCon & Financial Regulatory Intelligence Assistant")
    gr.Markdown(
        "**Production Hybrid RAG** • Hierarchical Parent-Child Indexing • Cross-Encoder Reranker • Citation Enforcement"
    )
    
    with gr.Row():
        # Left Configuration Column
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### ⚙️ Pipeline Configuration")
            top_k_slider = gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Top-K Rerank Results")
            
            ingest_btn = gr.Button("🔄 Run Document Ingestion", variant="secondary")
            ingest_status = gr.Textbox(label="Ingestion Status", interactive=False, lines=2)
            ingest_btn.click(fn=handle_ingestion, outputs=[ingest_status])
            
            gr.Markdown("---")
            gr.Markdown("### 📊 Sample 2026 Queries")
            sample_dropdown = gr.Dropdown(choices=SAMPLE_QUERIES, label="Select a sample query", value=None)
        
        # Right Main Interface Column
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Enter your Financial Controlling / Regulatory Question:",
                placeholder="e.g. What is the mandatory timeline for Form 8-K disclosure under SEC Item 1.05?",
                lines=3
            )
            sample_dropdown.change(fn=lambda s: s, inputs=[sample_dropdown], outputs=[query_input])
            
            search_btn = gr.Button("🔍 Search & Analyze", variant="primary", size="lg")
            
            with gr.Group():
                gr.Markdown("### 📝 Financial Controlling Synthesis")
                answer_output = gr.Markdown()
                meta_output = gr.Markdown()
                sources_output = gr.Markdown()
            
            with gr.Accordion("🔍 Deep Dive: Retrieved Child Chunks & Reranker Scores", open=False):
                chunks_output = gr.Markdown()
                
            with gr.Accordion("📖 Deep Dive: Full Parent Context Sent to LLM", open=False):
                context_output = gr.Code(language="markdown")
            
            search_btn.click(
                fn=handle_query,
                inputs=[query_input, top_k_slider],
                outputs=[answer_output, meta_output, sources_output, chunks_output, context_output]
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

