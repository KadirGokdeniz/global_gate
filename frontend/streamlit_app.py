import streamlit as st
import requests
import time
from datetime import datetime
import os
import json
import random

# Page configuration
st.set_page_config(
    page_title="Airline Policy Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with collapsed sidebar for cleaner look
)

# Modern CSS with improved design system
st.markdown("""
<style>
    /* Enhanced page background - fixed for full page coverage */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    }
    
    .main {
        background: transparent !important;
    }
    
    .main > div {
        padding-top: 2rem;
        background: transparent !important;
        min-height: 100vh;
        position: relative;
    }
    
    /* Body background fix for scroll */
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    }
    
    /* Animated background pattern that covers entire viewport */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        background-attachment: fixed !important;
    }
    
    .main > div::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(102, 126, 234, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(118, 75, 162, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 25%);
        background-size: 100px 100px, 150px 150px, 80px 80px;
        background-position: 0 0, 50px 50px, 25px 25px;
        animation: backgroundMove 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes backgroundMove {
        0%, 100% { transform: translateY(0px) translateX(0px); }
        33% { transform: translateY(-10px) translateX(10px); }
        66% { transform: translateY(10px) translateX(-5px); }
    }
    
    /* Advanced hero header with multiple animation layers */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 40px rgba(102, 126, 234, 0.3),
            0 10px 20px rgba(118, 75, 162, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
    }
    
    /* Floating particles effect */
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 20% 20%, rgba(255,255,255,0.3) 1px, transparent 1px),
            radial-gradient(circle at 80% 80%, rgba(255,255,255,0.2) 1px, transparent 1px),
            radial-gradient(circle at 40% 70%, rgba(255,255,255,0.15) 1px, transparent 1px),
            radial-gradient(circle at 60% 30%, rgba(255,255,255,0.25) 1px, transparent 1px);
        background-size: 100px 100px, 80px 80px, 120px 120px, 90px 90px;
        animation: floatParticles 15s linear infinite;
        pointer-events: none;
    }
    
    /* Shine wave effect */
    .hero-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255,255,255,0.2), 
            rgba(255,255,255,0.4), 
            rgba(255,255,255,0.2), 
            transparent
        );
        animation: shineWave 4s ease-in-out infinite;
        pointer-events: none;
    }
    
    .hero-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 
            0 2px 4px rgba(0,0,0,0.3),
            0 4px 8px rgba(0,0,0,0.2);
        animation: titleFloat 6s ease-in-out infinite;
        position: relative;
        z-index: 2;
    }
    
    .hero-header p {
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        position: relative;
        z-index: 2;
        animation: subtitleFloat 6s ease-in-out infinite 0.5s;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes floatParticles {
        0% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-20px) rotate(120deg); }
        66% { transform: translateY(10px) rotate(240deg); }
        100% { transform: translateY(0px) rotate(360deg); }
    }
    
    @keyframes shineWave {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-5px) scale(1.01); }
    }
    
    @keyframes subtitleFloat {
        0%, 100% { transform: translateY(0px); opacity: 0.95; }
        50% { transform: translateY(-3px); opacity: 1; }
    }
    
    /* Responsive airline buttons - improved */
    .airline-selection {
        margin: 1.5rem 0;
    }
    
    .airline-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    }
    
    @media (max-width: 768px) {
        .airline-grid {
            grid-template-columns: 1fr;
            gap: 0.75rem;
        }
    }
    
    /* Base airline button styling */
    .stButton > button {
        width: 100%;
        height: 60px !important;
        border-radius: 15px !important;
        border: 2px solid #e9ecef !important;
        background: rgba(248, 249, 250, 0.9) !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        color: #666 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        border-color: #ccc !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Active button styling - All Airlines (Blue) */
    .stButton > button[kind="primary"]:contains("🌍") {
        background: linear-gradient(135deg, #4285f4, #1976d2) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 6px 20px rgba(66, 133, 244, 0.4) !important;
    }
    
    /* Active button styling - Turkish Airlines (Red) */
    .stButton > button[kind="primary"]:contains("🇹🇷") {
        background: linear-gradient(135deg, #e53e3e, #c53030) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 6px 20px rgba(229, 62, 62, 0.4) !important;
    }
    
    /* Active button styling - Pegasus Airlines (Orange) */
    .stButton > button[kind="primary"]:contains("🔥") {
        background: linear-gradient(135deg, #ff6600, #e55100) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 6px 20px rgba(255, 102, 0, 0.4) !important;
    }
    
    .provider-card {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .provider-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .provider-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea20, #764ba220);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .provider-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .provider-name {
        font-weight: 600;
        font-size: 1.2rem;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .provider-desc {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Enhanced visual effects and micro-interactions */
    .question-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.1),
            0 4px 16px rgba(0,0,0,0.05),
            inset 0 1px 0 rgba(255,255,255,0.3);
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .question-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        animation: questionGlow 3s ease-in-out infinite;
    }
    
    @keyframes questionGlow {
        0%, 100% { left: -100%; opacity: 0; }
        50% { left: 100%; opacity: 1; }
    }
    
    .question-container h3 {
        color: #333;
        margin-bottom: 1rem;
        font-weight: 600;
        position: relative;
    }
    
    /* Floating action button style */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button[kind="primary"]:hover::before {
        left: 100%;
    }
    
    /* Progress indicators and loading states */
    .processing-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 1rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .loading-dots {
        display: flex;
        gap: 0.3rem;
    }
    
    .loading-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: loadingBounce 1.4s infinite ease-in-out both;
    }
    
    .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
    .loading-dots span:nth-child(3) { animation-delay: 0s; }
    
    @keyframes loadingBounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Improved mobile responsiveness */
    @media (max-width: 768px) {
        .hero-header {
            padding: 2rem 1rem;
            margin-bottom: 1rem;
        }
        
        .hero-header h1 {
            font-size: 2rem;
        }
        
        .hero-header p {
            font-size: 1rem;
        }
        
        .airline-button-container {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
        
        .airline-button {
            min-height: 60px;
            padding: 0.75rem;
        }
        
        .question-container,
        .response-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .main > div {
            padding-top: 1rem;
        }
    }
    
    /* Smooth scrolling and focus states */
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        outline: none;
    }
    
    /* Enhanced sidebar aesthetics */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .sidebar-section h4 {
        margin-bottom: 0.5rem;
        color: #333;
        font-weight: 600;
    }
    
    /* Chat history section - glassmorphism background */
    .chat-section {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.1),
            0 4px 16px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .chat-section h3 {
        margin-bottom: 1.5rem;
        color: #333;
        font-weight: 600;
    }
    
    /* Streamlit expander styling for chat history */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        margin-bottom: 0.5rem !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(102, 126, 234, 0.1) !important;
        backdrop-filter: blur(10px);
        padding: 1rem !important;
    }
    .response-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.1),
            0 4px 16px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-left: 4px solid #667eea;
        position: relative;
        overflow: hidden;
    }
    
    .response-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: cardGlow 3s ease infinite;
    }
    
    @keyframes cardGlow {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 0%; }
    }
    
    .response-meta {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .meta-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .source-badge {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        color: #495057;
    }
    
    .source-badge.turkish {
        background: linear-gradient(135deg, #dc3545, #e74c3c);
        color: white;
        border: none;
    }
    
    .source-badge.pegasus {
        background: linear-gradient(135deg, #fd7e14, #f39c12);
        color: white;
        border: none;
    }
    
    /* Airline filter pills */
    .filter-pills {
        display: flex;
        gap: 0.5rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .filter-pill {
        background: white;
        border: 2px solid #e9ecef;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .filter-pill:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    .filter-pill.active {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-color: transparent;
    }
    
    /* Enhanced Quick Question Cards */
    .quick-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .quick-card:hover {
        border-color: #667eea;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
    }
    
    .quick-card-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .quick-card-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #333;
        margin-bottom: 0.3rem;
    }
    
    .quick-card-desc {
        font-size: 0.8rem;
        color: #666;
        line-height: 1.3;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Sidebar improvements */
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .sidebar-section h4 {
        margin-bottom: 0.5rem;
        color: #333;
        font-weight: 600;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-header h1 {
            font-size: 2rem;
        }
        
        .provider-grid {
            grid-template-columns: 1fr;
        }
        
        .filter-pills {
            justify-content: center;
        }
        
        .question-container,
        .response-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
    
    /* Loading animations */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid #f3f3f3;
        border-radius: 50%;
        border-top: 2px solid #667eea;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {
        display: none;
    }
    
    .stDecoration {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced API Configuration with caching
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_api_urls():
    """Get multiple API URL options"""
    urls = [
        os.getenv('DEFAULT_API_URL', 'http://api:8000'),
        'http://localhost:8000',
        'http://127.0.0.1:8000',
    ]
    return urls

@st.cache_data(ttl=10)  # Cache for 10 seconds
def test_api_connection(api_url, timeout=2):
    """Test API connection with caching"""
    try:
        response = requests.get(f"{api_url}/health", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "status": "connected",
                "data": data,
                "url": api_url,
                "models_ready": data.get("models_preloaded", False)
            }
        else:
            return {
                "success": False,
                "status": f"http_error_{response.status_code}",
                "error": f"HTTP {response.status_code}",
                "url": api_url
            }
    except:
        return {
            "success": False,
            "status": "connection_error",
            "error": "Connection failed",
            "url": api_url
        }

def find_working_api():
    """Find first working API URL"""
    urls = get_api_urls()
    for url in urls:
        result = test_api_connection(url)
        if result["success"]:
            return result
    
    return {
        "success": False,
        "status": "all_failed",
        "error": "No API available",
        "tested_urls": urls
    }

# Initialize session state
def init_session_state():
    defaults = {
        'chat_history': [],
        'api_connection': None,
        'api_url': None,
        'selected_model': 'gpt-3.5-turbo',
        'selected_provider': 'OpenAI',
        'selected_airline': 'All Airlines',
        'current_question': '',
        'show_advanced': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_hero_header():
    """Enhanced hero header with floating particles animation"""
    st.markdown("""
    <div class="hero-header">
        <h1>✈️ Airline Policy Assistant</h1>
        <p>Get instant answers about airline policies powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

def display_provider_selection():
    """Modern AI provider selection"""
    st.markdown("### 🤖 Choose Your AI Assistant")
    
    providers = {
        'OpenAI': {
            'icon': '🧠',
            'name': 'OpenAI GPT',
            'desc': 'Fast and reliable responses',
            'models': ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4']
        },
        'Claude': {
            'icon': '🤖',
            'name': 'Anthropic Claude',
            'desc': 'Thoughtful and detailed analysis',
            'models': ['claude-3-haiku-20240307', 'claude-3-5-haiku-20241022', 'claude-sonnet-4-20250514']
        }
    }
    
    # Provider selection
    cols = st.columns(2)
    for i, (provider_key, provider_info) in enumerate(providers.items()):
        with cols[i]:
            is_selected = st.session_state.selected_provider == provider_key
            
            card_class = "provider-card selected" if is_selected else "provider-card"
            
            if st.button(
                f"{provider_info['icon']}\n{provider_info['name']}\n{provider_info['desc']}", 
                key=f"provider_{provider_key}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_provider = provider_key
                # Set default model for provider
                st.session_state.selected_model = provider_info['models'][0]
                st.rerun()
    
    # Model selection for selected provider
    if st.session_state.selected_provider in providers:
        current_provider = providers[st.session_state.selected_provider]
        
        with st.expander(f"⚙️ {current_provider['name']} Models", expanded=False):
            model = st.selectbox(
                "Select Model:",
                current_provider['models'],
                index=0 if st.session_state.selected_model not in current_provider['models'] 
                      else current_provider['models'].index(st.session_state.selected_model),
                key="model_selector"
            )
            st.session_state.selected_model = model

def display_airline_filters():
    """Modern airline filter pills"""
    st.markdown("### ✈️ Select Airlines")
    
    airlines = [
        ('All Airlines', '🌍', 'all'),
        ('Turkish Airlines', '🇹🇷', 'turkish'),
        ('Pegasus Airlines', '🔥', 'pegasus')
    ]
    
    cols = st.columns(len(airlines))
    for i, (name, icon, key) in enumerate(airlines):
        with cols[i]:
            is_active = st.session_state.selected_airline == name
            if st.button(
                f"{icon} {name}",
                key=f"airline_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.selected_airline = name
                st.rerun()

def display_quick_questions():
    """Enhanced quick question cards"""
    st.markdown("### 💡 Popular Questions")
    
    question_categories = {
        "✈️ Baggage Policies": [
            {"title": "Excess baggage fees comparison", "desc": "Compare Turkish Airlines vs Pegasus"},
            {"title": "Carry-on size limits", "desc": "International flight requirements"}
        ],
        "🐕 Pet Travel": [
            {"title": "Pet travel requirements", "desc": "Documents and carrier rules"},
            {"title": "Breed restrictions", "desc": "Which pets are allowed"}
        ],
        "🎵 Special Items": [
            {"title": "Musical instrument transport", "desc": "Size limits and special handling"},
            {"title": "Sports equipment rules", "desc": "Golf clubs, skiing gear etc."}
        ],
        "⚖️ Passenger Rights": [
            {"title": "Flight delay compensation", "desc": "Turkish airline policies"},
            {"title": "Cancellation rights", "desc": "Refund and rebooking options"}
        ]
    }
    
    for category, questions in question_categories.items():
        with st.expander(f"{category}", expanded=False):
            for i, q in enumerate(questions):
                if st.button(
                    f"**{q['title']}**\n{q['desc']}", 
                    key=f"cat_{category}_{i}",
                    use_container_width=True,
                    help=f"Ask about: {q['title']}"
                ):
                    st.session_state.current_question = q['title']
                    st.rerun()

def display_question_input():
    """Clean question input area without distracting elements"""
    st.markdown("### Ask Your Question")
    
    # Question input
    question = st.text_area(
        "",
        value=st.session_state.get('current_question', ''),
        placeholder="What would you like to know about airline policies?",
        height=100,
        key="question_input",
        label_visibility="collapsed"
    )
    
    # Simple action button
    ask_clicked = st.button(
        f"Ask {st.session_state.selected_provider}", 
        type="primary", 
        use_container_width=True,
        disabled=not question.strip()
    )
    
    return ask_clicked, question

def handle_question_optimized(question, api_url, model, provider, airline_filter):
    """Optimized question handling"""
    if not api_url:
        return {"success": False, "error": "No API connection"}
    
    # Determine endpoint
    endpoint = f"{api_url}/chat/claude" if provider == "Claude" else f"{api_url}/chat/openai"
    
    # Add airline context
    context_prefix = ""
    if "Turkish" in airline_filter:
        context_prefix = "Turkish Airlines: "
    elif "Pegasus" in airline_filter:
        context_prefix = "Pegasus Airlines: "
    
    enhanced_question = context_prefix + question
    
    try:
        with st.spinner(f"🤖 {provider} is analyzing airline policies..."):
            response = requests.get(
                endpoint,
                params={
                    "question": enhanced_question,
                    "max_results": 3,
                    "similarity_threshold": 0.4,
                    "model": model
                },
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return {
                    "success": True,
                    "answer": data["answer"],
                    "sources": data.get("retrieved_docs", []),
                    "model": data.get("model_used", model),
                    "provider": provider,
                    "retrieval_stats": data.get("retrieval_stats", {})
                }
            else:
                return {"success": False, "error": data.get("error", "Processing failed")}
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout (30s)"}
    except requests.exceptions.ConnectionError:
        st.session_state.api_connection = None
        return {"success": False, "error": "Connection lost"}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)[:50]}"}

def display_chat_history():
    """Display chat history with modern glassmorphism cards"""
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-section">
            <h3>Welcome!</h3>
            <p>Ask your first question about airline policies above.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="chat-section">
        <h3>Recent Conversations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Show last 3 conversations
    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
        with st.expander(f"Q: {chat['question'][:60]}...", expanded=(i == 0)):
            
            # Response metadata
            st.markdown(f"""
            <div class="response-meta">
                <span class="meta-badge">{chat['provider']} {chat['model']}</span>
                <span class="meta-badge">{len(chat.get('sources', []))} sources</span>
                <span class="meta-badge">{chat['timestamp'].strftime('%H:%M')}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Question and answer
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            
            # Sources
            if chat.get('sources'):
                st.markdown("**Sources:**")
                for doc in chat['sources'][:3]:
                    source = doc.get('source', 'Unknown')
                    similarity = doc.get('similarity_score', 0)
                    
                    badge_class = "source-badge"
                    if 'turkish' in source.lower() or any(kw in source.lower() for kw in ['checked_baggage', 'carry_on']):
                        badge_class += " turkish"
                    elif 'pegasus' in source.lower() or any(kw in source.lower() for kw in ['general_rules', 'baggage_allowance']):
                        badge_class += " pegasus"
                    
                    st.markdown(f'<span class="{badge_class}">{source} ({similarity:.1%})</span>', unsafe_allow_html=True)

def display_api_status():
    """Display API connection status"""
    if st.session_state.api_connection is None:
        st.session_state.api_connection = find_working_api()
    
    connection = st.session_state.api_connection
    
    if connection["success"]:
        st.session_state.api_url = connection["url"]
        models_ready = connection.get("models_ready", False)
        
        if models_ready:
            st.markdown("""
            <div class="status-indicator status-success">
                ⚡ API Connected (Models Ready)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-warning">
                ⏳ API Connected (Models Loading...)
            </div>
            """, unsafe_allow_html=True)
        return True
    else:
        st.markdown(f"""
        <div class="status-indicator status-error">
            ❌ API Connection Failed: {connection['error']}
        </div>
        """, unsafe_allow_html=True)
        return False

def display_sidebar():
    """Simplified sidebar - AI provider selection only"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h4>🔗 Connection Status</h4>
        </div>
        """, unsafe_allow_html=True)
        
        api_connected = display_api_status()
        
        if st.button("🔄 Reconnect", use_container_width=True):
            st.session_state.api_connection = None
            st.rerun()
        
        if api_connected:
            # AI Provider selection in sidebar
            st.markdown("""
            <div class="sidebar-section">
                <h4>🤖 AI Assistant</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Provider selection
            provider = st.selectbox(
                "Choose Provider:",
                ["OpenAI", "Claude"],
                index=0 if st.session_state.selected_provider == "OpenAI" else 1
            )
            st.session_state.selected_provider = provider
            
            # Model selection based on provider
            if provider == "Claude":
                models = ['claude-3-haiku-20240307', 'claude-3-5-haiku-20241022', 'claude-sonnet-4-20250514']
            else:
                models = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4']
            
            model = st.selectbox(
                "Model:",
                models,
                index=0 if st.session_state.selected_model not in models else models.index(st.session_state.selected_model)
            )
            st.session_state.selected_model = model
            
            # Performance section
            st.markdown("""
            <div class="sidebar-section">
                <h4>📊 Stats</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.chat_history:
                recent = st.session_state.chat_history[-5:]
                avg_sources = sum(len(chat.get('sources', [])) for chat in recent) / len(recent)
                st.metric("Avg Sources", f"{avg_sources:.1f}")
                st.metric("Total Queries", len(st.session_state.chat_history))
        
        # Quick Actions
        st.markdown("""
        <div class="sidebar-section">
            <h4>⚡ Actions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Advanced settings
        if st.checkbox("🔧 Advanced", value=False):
            st.slider("Max Results", 1, 10, 3, key="max_results")
            st.slider("Similarity Threshold", 0.1, 1.0, 0.4, step=0.1, key="similarity_threshold")

def main():
    """Main application"""
    init_session_state()
    display_hero_header()
    
    # Check API connection silently
    if st.session_state.api_connection is None:
        st.session_state.api_connection = find_working_api()
    
    if not st.session_state.api_connection["success"]:
        st.error("API service required for policy analysis")
        st.info("Please ensure the FastAPI service is running")
        return
    else:
        st.session_state.api_url = st.session_state.api_connection["url"]
    
    # Sidebar
    display_sidebar()
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Selectbox approach with color indicators
        st.markdown("### Choose Airlines")
        
        # Selectbox for airline selection
        airline_options = ["All Airlines", "Turkish Airlines Only", "Pegasus Airlines Only"]
        selected_airline = st.selectbox(
            "Select airline focus:",
            airline_options,
            index=airline_options.index(st.session_state.selected_airline) if st.session_state.selected_airline in airline_options else 0,
            key="airline_selectbox"
        )
        
        # Update session state
        st.session_state.selected_airline = selected_airline
        
        # Show colored feedback based on selection
        if selected_airline == "Turkish Airlines Only":
            st.error("🇹🇷 Turkish Airlines Focus - Queries will prioritize Turkish Airlines policies")
        elif selected_airline == "Pegasus Airlines Only":
            st.warning("🔥 Pegasus Airlines Focus - Queries will prioritize Pegasus Airlines policies")  
        else:
            st.success("🌍 All Airlines - Queries will search across all available airline policies")
        
        # Question input area
        ask_clicked, question = display_question_input()
        
        # Handle question
        if ask_clicked and question.strip():
            result = handle_question_optimized(
                question,
                st.session_state.api_url,
                st.session_state.selected_model,
                st.session_state.selected_provider,
                st.session_state.selected_airline
            )
            
            if result["success"]:
                # Add to history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "timestamp": datetime.now(),
                    "model": result["model"],
                    "provider": result["provider"],
                    "sources": result["sources"],
                    "airline_filter": st.session_state.selected_airline
                })
                
                st.success("Analysis complete!")
                st.rerun()
            else:
                st.error(f"{result['error']}")
        
        # Display chat history
        display_chat_history()
    
    with col2:
        # Quick questions
        display_quick_questions()

if __name__ == "__main__":
    main()