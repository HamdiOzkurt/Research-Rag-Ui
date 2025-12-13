"""
Minimal Claude-Style UI
Sade, beyaz, ortada input
"""
import streamlit as st
import asyncio
import sys
import os
import nest_asyncio

# Path ayarı
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.simple_agent import run_simple_research
from src.agents.main_agent import run_research
from src.agents.multi_agent_system import run_multi_agent_research
from src.config import settings

# Asyncio fix
nest_asyncio.apply()

# =============================================================================
# SAYFA AYARLARI
# =============================================================================
st.set_page_config(
    page_title="AI Research",
    page_icon="🔍",
    layout="centered",  # Ortada
    initial_sidebar_state="collapsed"  # Sidebar kapalı
)

# =============================================================================
# MİNİMAL CSS - CLAUDE TARZI
# =============================================================================
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global */
    * {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Ana container */
    .stApp {
        background: #ffffff;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Header temizle */
    header, #MainMenu, footer {
        visibility: hidden;
    }
    
    /* Sidebar gizle */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Chat container */
    .stChatMessage {
        background: transparent;
        border: none;
        padding: 1.5rem 0;
        margin: 0;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background: transparent;
    }
    
    .stChatMessage[data-testid="user-message"] [data-testid="chatAvatarIcon"] {
        background: #f3f4f6;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background: transparent;
    }
    
    /* Chat input */
    .stChatInput {
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
    }
    
    .stChatInput:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Markdown styling */
    .stMarkdown h1 {
        font-size: 1.875rem;
        font-weight: 600;
        color: #111827;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .stMarkdown h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    .stMarkdown h3 {
        font-size: 1.25rem;
        font-weight: 500;
        color: #374151;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .stMarkdown p {
        font-size: 1rem;
        line-height: 1.7;
        color: #374151;
        margin-bottom: 1rem;
    }
    
    .stMarkdown code {
        background: #f3f4f6;
        color: #1f2937;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9em;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }
    
    .stMarkdown pre {
        background: #1f2937;
        color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
        margin: 1rem 0;
    }
    
    .stMarkdown pre code {
        background: transparent;
        color: inherit;
        padding: 0;
    }
    
    .stMarkdown a {
        color: #3b82f6;
        text-decoration: none;
    }
    
    .stMarkdown a:hover {
        text-decoration: underline;
    }
    
    /* Status box */
    .stStatus {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    
    /* Button */
    .stButton button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #2563eb;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
    }
    
    /* Selectbox minimal */
    .stSelectbox {
        margin: 0;
    }
    
    .stSelectbox > div > div {
        border-color: #e5e7eb;
        border-radius: 8px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Hızlı Mod"  # Default

# =============================================================================
# RESEARCH FUNCTION
# =============================================================================
def do_research(query: str, mode: str) -> str:
    """Araştırma yap"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if mode == "Derin Araştırma":
            result = loop.run_until_complete(run_multi_agent_research(query, verbose=False))
        elif mode == "Standart Mod":
            result = loop.run_until_complete(run_research(query, verbose=False))
        else:  # Hızlı Mod
            result = loop.run_until_complete(run_simple_research(query, verbose=False))
        
        loop.close()
        return result
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return """⚠️ **API Limiti Aşıldı**

Gemini'nin günlük ücretsiz limiti (20 istek) doldu.

**Çözümler:**
- ⏰ 24 saat bekleyin
- 🔄 Hızlı Mod kullanın (daha az istek)
- 💳 Ücretli plan alın (çok ucuz)
- 🏠 Ollama kurun (ücretsiz, sınırsız)
"""
        return f"❌ Hata: {error_msg}"

# =============================================================================
# HEADER - MİNİMAL
# =============================================================================
st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)

# İlk açılışta karşılama
if not st.session_state.messages:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0 2rem 0;">
            <h1 style="font-size: 2.5rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">
                AI Research Assistant
            </h1>
            <p style="font-size: 1.125rem; color: #6b7280; margin-bottom: 2rem;">
                Web araştırması, kod örnekleri, Türkçe raporlar
            </p>
        </div>
        """, unsafe_allow_html=True)

# Mod seçici (minimal, üstte)
if not st.session_state.messages:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode = st.selectbox(
            "Araştırma Modu",
            ["Hızlı Mod", "Standart Mod", "Derin Araştırma"],
            index=0,
            help="Hızlı: 30s, Standart: 1-2dk, Derin: 3-5dk",
            label_visibility="collapsed"
        )
        st.session_state.mode = mode
        
        # Mod açıklaması
        mode_info = {
            "Hızlı Mod": "⚡ 30-60 saniye · Günlük kulanım için ideal",
            "Standart Mod": "🔍 1-2 dakika · Detaylı araştırma",
            "Derin Araştırma": "🧠 3-5 dakika · Çok detaylı + kod örnekleri"
        }
        st.markdown(f"""
        <div style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 0.5rem;">
            {mode_info[mode]}
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# CHAT INTERFACE
# =============================================================================

# Mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Bir şey sorun...", key="chat_input"):
    # User mesajı ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Assistant yanıtı
    with st.chat_message("assistant"):
        with st.status("Araştırılıyor...", expanded=False) as status:
            st.write(f"🔍 {st.session_state.mode} aktif")
            st.write("📡 Web araması yapılıyor...")
            
            response = do_research(prompt, st.session_state.mode)
            
            if response and not response.startswith("❌") and not response.startswith("⚠️"):
                status.update(label="✅ Tamamlandı", state="complete")
            else:
                status.update(label="⚠️ Hata", state="error")
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# =============================================================================
# FOOTER - MİNİMAL
# =============================================================================
if not st.session_state.messages:
    st.markdown('<div style="height: 4rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 0.875rem; padding: 2rem 0;">
        Powered by <strong>DeepAgents</strong> · Gemini 2.5 Flash
    </div>
    """, unsafe_allow_html=True)
else:
    # Chat başladıktan sonra mod seçici - floating
    st.markdown(f"""
    <div style="position: fixed; bottom: 80px; right: 20px; background: white; border: 1px solid #e5e7eb; 
                border-radius: 12px; padding: 0.75rem 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); z-index: 1000;">
        <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;">Mod</div>
        <div style="font-size: 0.875rem; font-weight: 500; color: #111827;">{st.session_state.mode}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clear button
    if st.button("🗑️ Temizle", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()
