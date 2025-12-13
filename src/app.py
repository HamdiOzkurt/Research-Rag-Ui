import streamlit as st
import asyncio
import sys
import os
import nest_asyncio

# Path Ayarı
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.main_agent import run_research
from src.config import settings

# Memory import (opsiyonel)
try:
    from src.memory import save_to_supabase, load_conversation_history
    MEMORY_ENABLED = True
except Exception as e:
    print(f"⚠️ Memory devre dışı: {e}")
    MEMORY_ENABLED = False

# Asyncio Fix
nest_asyncio.apply()

# =============================================================================
# SAYFA AYARLARI
# =============================================================================
st.set_page_config(
    page_title="DeepAgents Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODERN CSS - LIGHT/DARK UYUMLU
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root Variables - Light/Dark Uyumlu */
    :root {
        --primary-color: #6366f1;
        --primary-hover: #4f46e5;
        --success-color: #10b981;
        --error-color: #ef4444;
        --warning-color: #f59e0b;
    }
    
    /* Dark Mode */
    [data-theme="dark"] {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border-color: #475569;
    }
    
    /* Light Mode */
    [data-theme="light"] {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --border-color: #cbd5e1;
    }
    
    /* Genel Stil */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }
    
    /* Sidebar Title */
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--bg-tertiary);
        border-radius: 8px;
        font-size: 0.875rem;
        margin: 0.25rem 0;
        transition: all 0.2s;
    }
    
    .status-badge:hover {
        transform: translateX(2px);
    }
    
    /* Hero Section */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
        padding: 2rem;
    }
    
    .hero-title {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--primary-color), #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: clamp(1rem, 2vw, 1.25rem);
        color: var(--text-secondary);
        margin-bottom: 2.5rem;
        max-width: 600px;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Chat Input */
    .stChatInput > div > div > textarea {
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s;
    }
    
    .stChatInput > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border: 1px solid var(--border-color);
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border-left: 3px solid var(--primary-color);
    }
    
    /* Status Container */
    .stStatus {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Code Blocks */
    code {
        background: var(--bg-tertiary);
        color: var(--primary-color);
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.875rem;
    }
    
    /* Markdown */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .stMarkdown a {
        color: var(--primary-color);
        text-decoration: none;
        transition: opacity 0.2s;
    }
    
    .stMarkdown a:hover {
        opacity: 0.8;
        text-decoration: underline;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.3s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    if MEMORY_ENABLED:
        try:
            st.session_state.messages = load_conversation_history(n_messages=20)
            if st.session_state.messages:
                st.toast(f"✅ {len(st.session_state.messages)} mesaj yüklendi", icon="💾")
        except:
            st.session_state.messages = []
    else:
        st.session_state.messages = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model Info
    st.markdown("**🤖 Active Model**")
    try:
        model_name = settings.get_available_model()
    except:
        model_name = "Not configured"
    st.code(model_name, language="text")
    
    st.markdown("---")
    
    # System Status
    st.markdown("**📊 System Status**")
    
    # Gemini
    status_icon = "🟢" if settings.google_api_key else "🔴"
    st.markdown(f'<div class="status-badge">{status_icon} Gemini API</div>', unsafe_allow_html=True)
    
    # Firecrawl
    status_icon = "🟢" if settings.firecrawl_api_key else "🔴"
    st.markdown(f'<div class="status-badge">{status_icon} Firecrawl MCP</div>', unsafe_allow_html=True)
    
    # Tavily
    status_icon = "🟢" if hasattr(settings, 'tavily_api_key') and settings.tavily_api_key else "⚪"
    st.markdown(f'<div class="status-badge">{status_icon} Tavily Search</div>', unsafe_allow_html=True)
    
    # GitHub
    status_icon = "🟢" if hasattr(settings, 'github_token') and settings.github_token else "⚪"
    st.markdown(f'<div class="status-badge">{status_icon} GitHub MCP</div>', unsafe_allow_html=True)
    
    # Supabase
    status_icon = "🟢" if MEMORY_ENABLED else "⚪"
    st.markdown(f'<div class="status-badge">{status_icon} Supabase Memory</div>', unsafe_allow_html=True)
    
    # LangSmith
    status_icon = "🟢" if settings.langsmith_api_key else "⚪"
    st.markdown(f'<div class="status-badge">{status_icon} LangSmith Tracing</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear Chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.is_processing = False
        st.session_state.pending_query = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("**💡 Tips**")
    st.caption("• Use specific queries for better results")
    st.caption("• GitHub search works with code queries")
    st.caption("• All responses include source citations")

# =============================================================================
# RESEARCH FUNCTION
# =============================================================================
def do_research(query: str) -> str:
    """Araştırma yap"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_research(query, verbose=True))
        loop.close()
        return result
    except Exception as e:
        return f"❌ Hata: {str(e)}"

# =============================================================================
# MAIN APP
# =============================================================================

# Hero Screen (İlk açılış)
if not st.session_state.messages:
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🔬 DeepAgents Research</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">AI-powered research assistant with multi-source search, GitHub integration, and Turkish reporting</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.form("research_form", clear_on_submit=True):
        query = st.text_input(
            label="Research Query", 
            placeholder="e.g., Python pandas best practices, LangChain GitHub examples...", 
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("🚀 Start Research", use_container_width=True, type="primary")
        
        if submitted and query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.pending_query = query
            
            if MEMORY_ENABLED:
                save_to_supabase("user", query)
            
            st.rerun()

# Chat Screen
else:
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process pending query
    if st.session_state.pending_query and not st.session_state.is_processing:
        user_query = st.session_state.pending_query
        st.session_state.is_processing = True
        
        with st.chat_message("assistant"):
            status_container = st.status("🔍 Researching...", expanded=True)
            response_placeholder = st.empty()
            
            try:
                status_container.write("🤖 Initializing agents...")
                status_container.write("🌐 Searching multiple sources...")
                
                response = do_research(user_query)
                
                if response and not response.startswith("❌"):
                    status_container.update(label="✅ Research Complete", state="complete", expanded=False)
                    response_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    if MEMORY_ENABLED:
                        save_to_supabase("assistant", response)
                else:
                    status_container.update(label="❌ Error Occurred", state="error")
                    st.error(response)
                    
            except Exception as e:
                status_container.update(label="❌ System Error", state="error")
                st.error(f"Error: {str(e)}")
            
            finally:
                st.session_state.pending_query = None
                st.session_state.is_processing = False

    # Chat input
    if prompt := st.chat_input("Ask a follow-up question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_query = prompt
        
        if MEMORY_ENABLED:
            save_to_supabase("user", prompt)
        
        st.rerun()
