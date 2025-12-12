import streamlit as st
import asyncio
import sys
import os
import nest_asyncio

# Path Ayarı (Import hatalarını önlemek için)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.main_agent import run_research
from src.config import settings

# Asyncio Fix
nest_asyncio.apply()

# -----------------------------------------------------------------------------
# SAYFA AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepResearch",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# CSS & STİL (LangChain/LangGraph Tarzı Sade Tasarım)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Genel Font ve Renkler */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Input Alanını Özelleştirme */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border: 1px solid #4A4A4A;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
    }
    
    /* Chat Mesajları */
    .stChatMessage {
        background-color: transparent;
        border: none;
    }
    
    /* Kullanıcı Mesajı */
    .stChatMessage[data-testid="user-message"] {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* AI Mesajı */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: transparent;
        padding: 10px;
    }
    
    /* Hero Section (Ortadaki Büyük Input İçin) */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 2rem;
        background: -webkit-linear-gradient(45deg, #ffffff, #a0a0a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar Temizliği */
    section[data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #262730;
    }
    
    /* Status Container */
    .stStatus {
        background-color: #161920;
        border: 1px solid #262730;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")
    
    # Model Bilgisi
    st.markdown("**(LLM) Model**")
    st.code(settings.get_available_model(), language="text")
    
    st.markdown("---")
    
    # API Durumları (Minimalist)
    st.markdown("**System Status**")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("🟢" if settings.google_api_key else "🔴")
    with col2:
        st.markdown("Gemini API")
        
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("🟢" if settings.firecrawl_api_key else "🔴")
    with col2:
        st.markdown("Firecrawl MCP")
        
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("�" if settings.langsmith_api_key else "⚪")
    with col2:
        st.markdown("LangSmith Tracing")

# -----------------------------------------------------------------------------
# ANA UYGULAMA MANTIĞI
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DURUM 1: HİÇ MESAJ YOKSA (HERO EKRANI) ---
if not st.session_state.messages:
    # Sayfayı ortalamak için boşluklar
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">What do you want to know?</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Ortadaki Input
    # Not: Streamlit'te st.form kullanıyoruz ki enter'a basınca çalışsın
    with st.form("research_form", clear_on_submit=True):
        query = st.text_input(
            label="Research Query", 
            placeholder="e.g., Analysis of open source LLMs in 2024...", 
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Start Research", use_container_width=True, type="primary")
        
        if submitted and query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# --- DURUM 2: MESAJ VARSA (CHAT EKRANI) ---
else:
    # Geçmiş Mesajları Göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Eğer son mesaj kullanıcıdan geldiyse, AI yanıt üretmeli
    if st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            # Durum göstergesi
            status_container = st.status("Thinking...", expanded=True)
            response_placeholder = st.empty()
            
            try:
                # Araştırmayı Çalıştır
                status_container.write("Initializing agents...")
                status_container.write("Searching the web...")
                
                # Asenkron fonksiyonu çalıştır
                response = asyncio.run(run_research(user_query, verbose=True))
                
                if response and not response.startswith("❌"):
                    status_container.update(label="Research Complete", state="complete", expanded=False)
                    response_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    status_container.update(label="Error Occurred", state="error")
                    st.error(response)
                    
            except Exception as e:
                status_container.update(label="System Error", state="error")
                st.error(f"An error occurred: {str(e)}")

    # Alttaki Chat Input (Devam eden konuşmalar için)
    # Not: Burası sadece sohbet devam ederken görünür
    if prompt := st.chat_input("Ask a follow-up question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()
