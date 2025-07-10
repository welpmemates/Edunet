import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch
import networkx as nx
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Environmental AI Tools",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌍 Environmental AI Tools")
    st.markdown("A suite of advanced AI-powered tools for environmental analysis and content generation.")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("- Sentence Classifier\n- Image Generation\n- NER & Graph Mapping\n- Fill-in-the-Blank")
    st.markdown("---")
    st.info("Try the example inputs in each section!")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); font-family: 'Inter', sans-serif; }
    .main-title { font-size: 3rem; font-weight: 700; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 3rem; font-weight: 300; }
    .feature-card { background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%); border: 1px solid #4a5568; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); transition: all 0.3s ease; text-align: center; }
    .feature-card:hover { transform: translateY(-5px); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4); border-color: #667eea; }
    .feature-icon { font-size: 3rem; margin-bottom: 1rem; display: block; }
    .feature-title { font-size: 1.3rem; font-weight: 600; color: #f7fafc; margin-bottom: 0.5rem; }
    .feature-desc { color: #94a3b8; font-size: 0.9rem; line-height: 1.5; }
    .entity-tag { display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin: 0.25rem; }
    .result-card { background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1); }
    .result-item { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1rem; margin: 0.5rem 0; background: rgba(255, 255, 255, 0.03); border-radius: 8px; border-left: 3px solid #667eea; }
    .result-label { color: #f7fafc; font-weight: 500; }
    .result-score { color: #667eea; font-weight: 600; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; border-radius: 10px; color: white; font-weight: 600; padding: 0.75rem 2rem; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); }
    .stMarkdown, .stText { color: #f7fafc; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; color: #f7fafc; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: #667eea; box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2); }
    .js-plotly-plot { background: transparent !important; }
    .stProgress > div > div { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<h1 class="main-title">🌍 Environmental AI Tools</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered environmental analysis and content generation</p>', unsafe_allow_html=True)

# --- FEATURE CARDS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🤖</span>
        <h3 class="feature-title">AI-Powered</h3>
        <p class="feature-desc">Advanced deep learning models for environmental analysis</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <h3 class="feature-title">Fast Analysis</h3>
        <p class="feature-desc">Real-time classification and processing</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🎯</span>
        <h3 class="feature-title">High Accuracy</h3>
        <p class="feature-desc">Precise environmental content detection</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🌐</span>
        <h3 class="feature-title">Global Coverage</h3>
        <p class="feature-desc">Works with content worldwide</p>
    </div>
    """, unsafe_allow_html=True)

# --- SESSION STATE: LOAD MODELS ---
if "classifier" not in st.session_state:
    with st.spinner("Loading AI models..."):
        st.session_state.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        st.session_state.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        st.session_state.fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# --- ENVIRONMENTAL CATEGORIES ---
CATEGORIES = [
    "Climate Change",
    "Wildlife Conservation",
    "Pollution Control",
    "Renewable Energy",
    "Environmental Policy"
]

# --- TOOL 1: SENTENCE CLASSIFIER ---
with st.expander("🌍 Environmental Sentence Classifier", expanded=True):
    st.caption("Classify a sentence into environmental categories. Example: 'Solar panels are becoming more efficient and affordable.'")
    sentence = st.text_input("Enter a sentence to classify:", value="Solar panels are becoming more efficient and affordable.")
    if st.button("Classify Sentence"):
        with st.spinner('Classifying sentence...'):
            result = st.session_state.classifier(sentence, CATEGORIES)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("**Classification Results:**")
        for label, score in zip(result["labels"], result["scores"]):
            st.markdown(f"""
            <div class="result-item">
                <span class="result-label">{label}</span>
                <span class="result-score">{score:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # Visualization
        fig = px.bar(
            x=result["scores"], 
            y=result["labels"],
            orientation='h',
            color=result["scores"],
            color_continuous_scale='viridis',
            title="Classification Confidence"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TOOL 2: IMAGE GENERATION ---
@st.cache_resource
def load_sd_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe

with st.expander("🖼️ Image Generation", expanded=True):
    st.caption("Describe an environmental scene to generate an image. Example: 'A beautiful forest with clean air and wildlife.'")
    prompt = st.text_input("Enter a prompt for image generation:", value="A beautiful forest with clean air and wildlife.")
    if st.button("Generate Image"):
        pipe = load_sd_pipeline()
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
        st.image(image, caption=f"Generated: {prompt}", use_container_width=True)

# --- TOOL 3: NER & GRAPH MAPPING ---
with st.expander("🔍 Named Entity Recognition & Graph Mapping", expanded=True):
    st.caption("Extract entities from text and visualize their relationships. Example: 'The Amazon rainforest in Brazil is home to countless species and plays a crucial role in global climate regulation.'")
    ner_text = st.text_area("Enter text for NER and graph mapping:", value="The Amazon rainforest in Brazil is home to countless species and plays a crucial role in global climate regulation.")
    if st.button("Extract Entities and Show Graph"):
        with st.spinner('Extracting entities...'):
            entities = st.session_state.ner_pipeline(ner_text)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("**Recognized Entities:**")
        entity_html = ""
        for ent in entities:
            entity_html += f'<span class="entity-tag">{ent["entity_group"]}: {ent["word"]}</span>'
        st.markdown(entity_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # Create and display graph
        if entities:
            G = nx.Graph()
            entity_words = [ent['word'] for ent in entities]
            for word in entity_words:
                G.add_node(word)
            for i in range(len(entity_words)):
                for j in range(i + 1, len(entity_words)):
                    G.add_edge(entity_words[i], entity_words[j])
            pos = nx.spring_layout(G)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            node_x, node_y, node_text = [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=50,
                    color='#667eea',
                    line=dict(width=2, color='white')
                )
            )
            fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Entity Relationship Graph',
                    title_font_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Entities and their relationships",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="white", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                ))
            st.plotly_chart(fig, use_container_width=True)

# --- TOOL 4: FILL-IN-THE-BLANK ---
with st.expander("💭 Fill-in-the-Blank (Masked Language Modeling)", expanded=True):
    st.caption("Enter a sentence with [MASK] to predict possible words. Example: 'Climate change is a [MASK] issue.'")
    mask_input = st.text_input("Enter a sentence with [MASK]:", value="Climate change is a [MASK] issue.")
    if st.button("Predict Mask"):
        if '[MASK]' in mask_input:
            with st.spinner('Predicting mask...'):
                results = st.session_state.fill_mask(mask_input)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("**Predictions:**")
            for res in results:
                st.markdown(f"""
                <div class="result-item">
                    <span class="result-label">{res['sequence']}</span>
                    <span class="result-score">{res['score']:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Please include [MASK] in your sentence!")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; margin-top: 2rem;">
    <p>🌍 Environmental AI Tools - Powered by Advanced Machine Learning</p>
    <p style="font-size: 0.8rem;">Built with Streamlit, Transformers, and Plotly</p>
</div>
""", unsafe_allow_html=True)
