# streamlit_app.py - Complete Multilingual Version - FIXED

from typing import Optional
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
    initial_sidebar_state="collapsed"
)

# MULTILINGUAL SUPPORT - Translation Dictionary
TRANSLATIONS = {
    "en": {
        "title": "Airline Policy Assistant",
        "subtitle": "Get instant answers about airline policies powered by AI",
        "ask_question": "Ask Your Question",
        "question_placeholder": "What would you like to know about airline policies?",
        "choose_provider": "Choose Provider:",
        "choose_model": "Model:",
        "choose_airline": "Choose Airlines",
        "all_airlines": "All Airlines",
        "turkish_airlines_only": "Turkish Airlines Only", 
        "pegasus_only": "Pegasus Airlines Only",
        "ask_button": "Ask",
        "recent_conversations": "Recent Conversations",
        "popular_questions": "Popular Questions",
        "baggage_policies": "Baggage Policies",
        "pet_travel": "Pet Travel",
        "special_items": "Special Items", 
        "passenger_rights": "Passenger Rights",
        "feedback": "Feedback",
        "helpful": "👍 Helpful",
        "not_helpful": "👎 Not Helpful",
        "too_slow": "⏱️ Too Slow",
        "wrong_info": "❌ Wrong Info",
        "session_stats": "Session Stats",
        "total_queries": "Total Queries",
        "satisfaction": "Satisfaction", 
        "clear_history": "🗑️ Clear All History",
        "api_connected": "⚡ API Connected",
        "api_failed": "❌ API Connection Failed",
        "reconnect": "🔄 Reconnect",
        "analyzing": "is analyzing airline policies...",
        "analysis_complete": "✅ Analysis complete!",
        "connection_lost": "Connection lost",
        "request_timeout": "Request timeout",
        "welcome_message": "Welcome to AI Assistant!",
        "welcome_description": "Ask your first question about airline policies above.",
        "features": "Features:",
        "smart_search": "✅ Smart Policy Search",
        "fast_response": "⚡ Fast Response Times", 
        "quality_tracking": "📊 Quality Tracking",
        "satisfaction_tracking": "🎯 User Satisfaction",
        "language_selector": "Language:",
        "excess_baggage_comparison": "Excess baggage fees comparison",
        "thy_vs_pegasus": "Turkish Airlines vs Pegasus",
        "carryon_limits": "Carry-on size limits", 
        "international_requirements": "International flight requirements",
        "pet_requirements": "Pet travel requirements",
        "docs_and_carriers": "Documents and carrier rules",
        "breed_restrictions": "Breed restrictions",
        "allowed_pets": "Which pets are allowed",
        "instrument_transport": "Musical instrument transport",
        "size_limits_handling": "Size limits and special handling",
        "sports_equipment": "Sports equipment rules",
        "golf_skiing_etc": "Golf clubs, skiing gear etc.",
        "delay_compensation": "Flight delay compensation", 
        "turkish_policies": "Turkish airline policies",
        "cancellation_rights": "Cancellation rights",
        "refund_rebooking": "Refund and rebooking options",
        "sources": "Sources:",
        "sources_retrieved": "Sources Retrieved",
        "avg_similarity": "Avg Similarity", 
        "context_quality": "Context Quality",
        "session_id": "Session ID:",
        "you_found_helpful": "✅ You found this helpful",
        "you_marked_not_helpful": "⚠️ You marked this as not helpful",
        "you_reported_slow": "⏱️ You reported this was too slow", 
        "you_reported_incorrect": "❌ You reported incorrect information",
        "feedback_recorded": "Feedback recorded",
        "thanks_feedback": "Thanks for your feedback!",
        "thanks_review": "Thanks, we'll review this!",
        "work_on_speed": "We'll work on speed!",
        "airline_focus_thy": "🇹🇷 Turkish Airlines Focus - Queries will prioritize Turkish Airlines policies",
        "airline_focus_pegasus": "✈️ Pegasus Airlines Focus - Queries will prioritize Pegasus Airlines policies",
        "airline_focus_all": "🌍 All Airlines - Queries will search across all available airline policies"
    },
    "tr": {
        "title": "Havayolu Politika Asistanı", 
        "subtitle": "Yapay zeka destekli havayolu politikaları danışmanınız",
        "ask_question": "Sorunuzu Sorun",
        "question_placeholder": "Havayolu politikaları hakkında merak ettiklerinizi yazın...",
        "choose_provider": "AI Sağlayıcısı Seçin:",
        "choose_model": "Model:",
        "choose_airline": "Havayolu Seçin",
        "all_airlines": "Tüm Havayolları",
        "turkish_airlines_only": "Sadece Türk Hava Yolları",
        "pegasus_only": "Sadece Pegasus Hava Yolları", 
        "ask_button": "AI Asistanına Sor",
        "recent_conversations": "Son Konuşmalar",
        "popular_questions": "💡 Popüler Sorular", 
        "baggage_policies": "✈️ Bagaj Politikaları",
        "pet_travel": "🐕 Evcil Hayvan Seyahati",
        "special_items": "🎵 Özel Eşyalar",
        "passenger_rights": "⚖️ Yolcu Hakları",
        "feedback": "📝 Geri Bildirim",
        "helpful": "👍 Yardımcı Oldu",
        "not_helpful": "👎 Yardımcı Olmadı",
        "too_slow": "⏱️ Çok Yavaş", 
        "wrong_info": "❌ Yanlış Bilgi",
        "session_stats": "📊 Oturum İstatistikleri",
        "total_queries": "Toplam Sorgu",
        "satisfaction": "Memnuniyet",
        "clear_history": "🗑️ Geçmişi Temizle",
        "api_connected": "⚡ API Bağlandı",
        "api_failed": "❌ API Bağlantısı Başarısız",
        "reconnect": "🔄 Yeniden Bağlan",
        "analyzing": "havayolu politikalarını analiz ediyor...",
        "analysis_complete": "✅ Analiz tamamlandı!",
        "connection_lost": "Bağlantı kesildi", 
        "request_timeout": "İstek zaman aşımı",
        "welcome_message": "Havayolu Politikaları AI Asistanına Hoş Geldiniz!",
        "welcome_description": "Yukarıdan havayolu politikaları hakkında ilk sorunuzu sorun.",
        "features": "Özellikler:",
        "smart_search": "✅ Akıllı Politika Arama",
        "fast_response": "⚡ Hızlı Yanıt Süreleri",
        "quality_tracking": "📊 Kalite Takibi", 
        "satisfaction_tracking": "🎯 Memnuniyet Takibi",
        "language_selector": "Dil:",
        "excess_baggage_comparison": "Fazla bagaj ücretleri karşılaştırması",
        "thy_vs_pegasus": "THY vs Pegasus karşılaştırması",
        "carryon_limits": "El bagajı boyut sınırları",
        "international_requirements": "Uluslararası uçuş gereksinimleri",
        "pet_requirements": "Evcil hayvan seyahat gereksinimleri",
        "docs_and_carriers": "Belgeler ve taşıyıcı kuralları",
        "breed_restrictions": "Cins kısıtlamaları", 
        "allowed_pets": "Hangi hayvanlar izinli",
        "instrument_transport": "Müzik aleti taşıma",
        "size_limits_handling": "Boyut sınırları ve özel işlemler",
        "sports_equipment": "Spor malzemesi kuralları",
        "golf_skiing_etc": "Golf sopası, kayak ekipmanı vb.",
        "delay_compensation": "Uçuş gecikme tazminatı",
        "turkish_policies": "Türk havayolu politikaları",
        "cancellation_rights": "İptal hakları",
        "refund_rebooking": "İade ve yeniden rezervasyon seçenekleri",
        "sources": "Kaynaklar:",
        "sources_retrieved": "Bulunan Kaynak",
        "avg_similarity": "Ort. Benzerlik",
        "context_quality": "İçerik Kalitesi", 
        "session_id": "Oturum ID:",
        "you_found_helpful": "✅ Bu yanıtın yardımcı olduğunu belirttiniz",
        "you_marked_not_helpful": "⚠️ Bu yanıtın yardımcı olmadığını belirttiniz",
        "you_reported_slow": "⏱️ Bu yanıtın çok yavaş olduğunu belirttiniz",
        "you_reported_incorrect": "❌ Yanlış bilgi olduğunu bildirdiniz",
        "feedback_recorded": "Geri bildirim kaydedildi",
        "thanks_feedback": "Geri bildiriminiz için teşekkürler!",
        "thanks_review": "Teşekkürler, bu durumu inceleyeceğiz!", 
        "work_on_speed": "Hızda iyileştirme için çalışacağız!",
        "airline_focus_thy": "🇹🇷 Türk Hava Yolları Odağı - Sorgular THY politikalarını önceleyecek",
        "airline_focus_pegasus": "✈️ Pegasus Hava Yolları Odağı - Sorgular Pegasus politikalarını önceleyecek", 
        "airline_focus_all": "🌍 Tüm Havayolları - Sorgular mevcut tüm havayolu politikalarında arama yapacak"
    }
}

def get_text(key: str, lang: str = None) -> str:
    """Get translated text with proper error handling"""
    if lang is None:
        lang = st.session_state.get('language', 'en')
    return TRANSLATIONS.get(lang, {}).get(key, key)

# Enhanced CSS with language selector support
st.markdown("""
<style>
    /* Enhanced page background */
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
    
    /* Language selector styling */
    .language-selector {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .language-flag {
        font-size: 1.5rem;
        margin: 0 0.25rem;
        cursor: pointer;
        padding: 0.25rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        display: inline-block;
    }
    
    .language-flag:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: scale(1.1);
    }
    
    .language-flag.active {
        background: rgba(102, 126, 234, 0.2);
        transform: scale(1.05);
    }
    
    /* Hero header with language support */
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
    
    .hero-header-tr {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 50%, #e74c3c 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 40px rgba(231, 76, 60, 0.3),
            0 10px 20px rgba(192, 57, 43, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
    }
    
    /* Floating particles animation in hero */
    .hero-header::before, .hero-header-tr::before {
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
    
    .hero-header h1, .hero-header-tr h1 {
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
    
    .hero-header p, .hero-header-tr p {
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
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-5px) scale(1.01); }
    }
    
    @keyframes subtitleFloat {
        0%, 100% { transform: translateY(0px); opacity: 0.95; }
        50% { transform: translateY(-3px); opacity: 1; }
    }
    
    /* Chat section */
    .chat-section {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
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
    
    /* Feedback section */
    .feedback-section {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .feedback-section h4 {
        color: #495057;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Response metadata */
    .response-meta {
        margin: 1rem 0;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    
    .meta-badge {
        background: #e9ecef;
        color: #495057;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Session info */
    .session-info {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.85rem;
        color: #4a5568;
        border-left: 3px solid #667eea;
    }
    
    .session-info-tr {
        background: rgba(231, 76, 60, 0.1);
        border-left: 3px solid #e74c3c;
    }
    
    /* Hidden button styles */
    .lang-button-hidden {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Multilingual quick questions
QUICK_QUESTIONS = {
    "en": {
        "Baggage Policies": [
            {"title": "Excess baggage fees comparison", "desc": "Turkish Airlines vs Pegasus"},
            {"title": "Carry-on size limits", "desc": "International flight requirements"}
        ],
        "Pet Travel": [
            {"title": "Pet travel requirements", "desc": "Documents and carrier rules"},
            {"title": "Breed restrictions", "desc": "Which pets are allowed"}
        ],
        "Special Items": [
            {"title": "Musical instrument transport", "desc": "Size limits and special handling"},
            {"title": "Sports equipment rules", "desc": "Golf clubs, skiing gear etc."}
        ],
        "Passenger Rights": [
            {"title": "Flight delay compensation", "desc": "Turkish airline policies"},
            {"title": "Cancellation rights", "desc": "Refund and rebooking options"}
        ]
    },
    "tr": {
        "Bagaj Politikaları": [
            {"title": "Fazla bagaj ücretleri karşılaştırması", "desc": "THY vs Pegasus karşılaştırması"},
            {"title": "El bagajı boyut sınırları", "desc": "Uluslararası uçuş gereksinimleri"}
        ],
        "Evcil Hayvan Seyahati": [
            {"title": "Evcil hayvan seyahat gereksinimleri", "desc": "Belgeler ve taşıyıcı kuralları"},
            {"title": "Cins kısıtlamaları", "desc": "Hangi hayvanlar izinli"}
        ],
        "Özel Eşyalar": [
            {"title": "Müzik aleti taşıma", "desc": "Boyut sınırları ve özel işlemler"},
            {"title": "Spor malzemesi kuralları", "desc": "Golf sopası, kayak ekipmanı vb."}
        ],
        "Yolcu Hakları": [
            {"title": "Uçuş gecikme tazminatı", "desc": "Türk havayolu politikaları"},
            {"title": "İptal hakları", "desc": "İade ve yeniden rezervasyon seçenekleri"}
        ]
    }
}

# Language-aware airline mapping
def get_airline_mapping():
    """Get airline mapping based on current language"""
    try:
        lang = st.session_state.get('language', 'en')
        return {
            get_text("all_airlines", lang): None,
            get_text("turkish_airlines_only", lang): "turkish_airlines",
            get_text("pegasus_only", lang): "pegasus"
        }
    except:
        # Fallback mapping
        return {
            "All Airlines": None,
            "Turkish Airlines Only": "turkish_airlines", 
            "Pegasus Airlines Only": "pegasus",
            "Tüm Havayolları": None,
            "Sadece Türk Hava Yolları": "turkish_airlines",
            "Sadece Pegasus Hava Yolları": "pegasus"
        }

def display_language_selector():
    """Display language selector - Simple and functional approach"""
    current_lang = st.session_state.get('language', 'en')
    
    # Simple header with language buttons - UPDATED: moved to far right
    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
    
    with col1:
        pass  # boş alan
    
    with col4:
        if st.button("🇺🇸 EN", key="lang_en_btn", 
                    type="primary" if current_lang == 'en' else "secondary"):
            st.session_state.language = 'en'
            # Update airline selection to match new language
            if 'selected_airline' in st.session_state:
                airline_mapping = get_airline_mapping()
                st.session_state.selected_airline = list(airline_mapping.keys())[0]
            st.rerun()
    
    with col5:
        if st.button("🇹🇷 TR", key="lang_tr_btn",
                    type="primary" if current_lang == 'tr' else "secondary"):
            st.session_state.language = 'tr'
            # Update airline selection to match new language
            if 'selected_airline' in st.session_state:
                airline_mapping = get_airline_mapping()
                st.session_state.selected_airline = list(airline_mapping.keys())[0]
            st.rerun()
    
    st.markdown("---")  # Separator

# API Configuration with caching
@st.cache_data(ttl=30)
def get_api_urls():
    """Get multiple API URL options"""
    urls = [
        os.getenv('DEFAULT_API_URL', 'http://api:8000'),
        'http://localhost:8000',
        'http://127.0.0.1:8000',
    ]
    return urls

@st.cache_data(ttl=10)
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
                "models_ready": data.get("features", {}).get("models_preloaded", False)
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

# FIXED Session state initialization with language support
def init_session_state():
    """Initialize session state with proper error handling"""
    defaults = {
        'chat_history': [],
        'api_connection': None,
        'api_url': None,
        'selected_model': 'gpt-3.5-turbo',
        'selected_provider': 'OpenAI',
        'selected_airline': '',  # Will be set after language is determined
        'current_question': '',
        'show_advanced': False,
        'feedback_given': {},
        'session_tracking': {},
        'language': 'en'  # Default language is English
    }
    
    # Initialize all defaults first
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Set default airline selection based on language - FIXED LOGIC
    if st.session_state.get('selected_airline') is None or st.session_state.get('selected_airline') == '':
        try:
            # Get airline mapping safely
            airline_mapping = get_airline_mapping()
            # Set to first airline option (All Airlines)
            st.session_state.selected_airline = list(airline_mapping.keys())[0]
        except Exception as e:
            # Ultimate fallback
            lang = st.session_state.get('language', 'en')
            fallback = "All Airlines" if lang == 'en' else "Tüm Havayolları"
            st.session_state.selected_airline = fallback

def display_hero_header():
    """Language-aware hero header"""
    lang = st.session_state.get('language', 'en')
    header_class = "hero-header-tr" if lang == 'tr' else "hero-header"
    
    st.markdown(f"""
    <div class="{header_class}">
        <h1>✈️ {get_text('title')}</h1>
        <p>{get_text('subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

def display_question_input():
    """Language-aware question input area"""
    st.markdown(f"### {get_text('ask_question')}")
    
    question = st.text_area(
        "",
        value=st.session_state.get('current_question', ''),
        placeholder=get_text('question_placeholder'),
        height=100,
        key="question_input"
    )
    
    provider = st.session_state.selected_provider
    ask_clicked = st.button(
        f"{get_text('ask_button')} {provider}", 
        type="primary", 
        use_container_width=True,
        disabled=not question.strip()
    )
    
    return ask_clicked, question

def handle_question_optimized(question, api_url, model, provider, airline_selection):
    """Language-aware question handling"""
    lang = st.session_state.get('language', 'en')
    
    if not api_url:
        return {"success": False, "error": "No API connection"}
    
    airline_preference = get_airline_mapping().get(airline_selection)
    
    endpoint = f"{api_url}/chat/claude" if provider == "Claude" else f"{api_url}/chat/openai"
    
    try:
        with st.spinner(f"🤖 {provider} {get_text('analyzing')}"):
            params = {
                "question": question,
                "max_results": 3,
                "similarity_threshold": 0.4,
                "model": model,
                "language": lang  # IMPORTANT: Send language parameter
            }
            
            # Add airline preference if specified
            if airline_preference:
                params["airline_preference"] = airline_preference
            
            response = requests.get(endpoint, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return {
                    "success": True,
                    "session_id": data.get("session_id"),
                    "answer": data["answer"],
                    "sources": data.get("sources", []),
                    "model": data.get("model_used", model),
                    "provider": provider,
                    "stats": data.get("stats", {}),
                    "preference_stats": data.get("preference_stats", {}),
                    "airline_preference": data.get("airline_preference"),
                    "performance": data.get("performance", {}),
                    "language": data.get("language", lang)
                }
            else:
                return {"success": False, "error": data.get("error", "Processing failed")}
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"{get_text('request_timeout')} (30s)"}
    except requests.exceptions.ConnectionError:
        st.session_state.api_connection = None
        return {"success": False, "error": get_text('connection_lost')}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)[:50]}"}

def display_airline_selection():
    """Language-aware airline selection"""
    st.markdown(f"### {get_text('choose_airline')}")
    
    try:
        airline_mapping = get_airline_mapping()
        airline_options = list(airline_mapping.keys())
        
        # Ensure current selection is valid for current language
        current_selection = st.session_state.get('selected_airline')
        if current_selection not in airline_options:
            current_selection = airline_options[0]
            st.session_state.selected_airline = current_selection
        
        selected_airline = st.selectbox(
            "Select airline focus:",
            airline_options,
            index=airline_options.index(current_selection) if current_selection in airline_options else 0,
            key="airline_selectbox"
        )
        
        st.session_state.selected_airline = selected_airline
        
        # Airline feedback with language support
        if selected_airline in [get_text('turkish_airlines_only'), "Turkish Airlines Only", "Sadece Türk Hava Yolları"]:
            st.error(get_text('airline_focus_thy'))
        elif selected_airline in [get_text('pegasus_only'), "Pegasus Airlines Only", "Sadece Pegasus Hava Yolları"]:
            st.warning(get_text('airline_focus_pegasus'))
        else:
            st.success(get_text('airline_focus_all'))
            
    except Exception as e:
        st.error(f"Airline selection error: {e}")
        # Set fallback
        lang = st.session_state.get('language', 'en')
        fallback = "All Airlines" if lang == 'en' else "Tüm Havayolları"
        st.session_state.selected_airline = fallback

def display_api_status():
    """Language-aware API status display"""
    if st.session_state.api_connection is None:
        st.session_state.api_connection = find_working_api()
    
    connection = st.session_state.api_connection
    
    if connection["success"]:
        st.session_state.api_url = connection["url"]
        models_ready = connection.get("models_ready", False)
        
        if models_ready:
            st.markdown(f"""
            <div class="status-indicator status-success">
                {get_text('api_connected')} (Models Ready)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-indicator status-warning">
                {get_text('api_connected')} (Loading...)
            </div>
            """, unsafe_allow_html=True)
        return True
    else:
        st.markdown(f"""
        <div class="status-indicator status-error">
            {get_text('api_failed')}: {connection['error']}
        </div>
        """, unsafe_allow_html=True)
        return False

def send_feedback(chat_item, feedback_type):
    """Send user feedback to API"""
    try:
        api_url = st.session_state.get('api_url')
        if not api_url:
            return
        
        feedback_data = {
            "question": chat_item['question'],
            "answer": chat_item['answer'],
            "feedback_type": feedback_type,
            "provider": chat_item['provider'],
            "model": chat_item['model']
        }
        
        response = requests.post(
            f"{api_url}/feedback",
            json=feedback_data,
            timeout=5
        )
        
        if response.status_code == 200:
            st.session_state.feedback_sent = True
        
    except Exception as e:
        st.error(f"Feedback could not be sent: {e}")

def display_chat_history():
    """Language-aware chat history display"""
    lang = st.session_state.get('language', 'en')
    
    if not st.session_state.chat_history:
        st.markdown(f"""
        <div class="chat-section">
            <h3>{get_text('welcome_message')}</h3>
            <p>{get_text('welcome_description')} {get_text('features')}</p>
            <ul>
                <li>{get_text('smart_search')}</li>
                <li>{get_text('fast_response')}</li>
                <li>{get_text('quality_tracking')}</li>
                <li>{get_text('satisfaction_tracking')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div class="chat-section">
        <h3>{get_text('recent_conversations')}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Show last 3 conversations
    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
        chat_id = f"{chat['timestamp'].strftime('%Y%m%d_%H%M%S')}_{i}"
        session_id = chat.get('session_id', 'unknown')
        
        with st.expander(f"Q: {chat['question'][:60]}...", expanded=(i == 0)):
            
            # Metadata
            st.markdown(f"""
            <div class="response-meta">
                <span class="meta-badge">{chat['provider']} {chat.get('model', 'Unknown')}</span>
                <span class="meta-badge">{len(chat.get('sources', []))} sources</span>
                <span class="meta-badge">{chat['timestamp'].strftime('%H:%M')}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Question and answer
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            
            # Session ID info
            if session_id != 'unknown':
                session_class = "session-info-tr" if lang == 'tr' else "session-info"
                st.markdown(f"""
                <div class="{session_class}">
                    {get_text('session_id')} {session_id[:16]}...
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Language-aware Feedback Section
            st.markdown(f"""
            <div class="feedback-section">
                <h4>{get_text('feedback')}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if feedback already given for this chat
            feedback_given = st.session_state.feedback_given.get(chat_id, None)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                button_type = "primary" if feedback_given == "helpful" else "secondary"
                disabled = feedback_given is not None and feedback_given != "helpful"
                
                if st.button(
                    get_text('helpful'), 
                    key=f"feedback_{chat_id}_helpful",
                    help="This answer was helpful",
                    type=button_type,
                    disabled=disabled
                ):
                    if feedback_given is None:
                        send_feedback(chat, "helpful")
                        st.session_state.feedback_given[chat_id] = "helpful"
                        st.success(get_text('thanks_feedback'))
                        st.rerun()
            
            with col2:
                button_type = "primary" if feedback_given == "not_helpful" else "secondary"
                disabled = feedback_given is not None and feedback_given != "not_helpful"
                
                if st.button(
                    get_text('not_helpful'), 
                    key=f"feedback_{chat_id}_not_helpful",
                    help="This answer was not helpful",
                    type=button_type,
                    disabled=disabled
                ):
                    if feedback_given is None:
                        send_feedback(chat, "not_helpful")
                        st.session_state.feedback_given[chat_id] = "not_helpful"
                        st.info(get_text('thanks_feedback'))
                        st.rerun()
            
            with col3:
                button_type = "primary" if feedback_given == "too_slow" else "secondary"
                disabled = feedback_given is not None and feedback_given != "too_slow"
                
                if st.button(
                    get_text('too_slow'), 
                    key=f"feedback_{chat_id}_too_slow",
                    help="Response was too slow",
                    type=button_type,
                    disabled=disabled
                ):
                    if feedback_given is None:
                        send_feedback(chat, "too_slow")
                        st.session_state.feedback_given[chat_id] = "too_slow"
                        st.warning(get_text('work_on_speed'))
                        st.rerun()
            
            with col4:
                button_type = "primary" if feedback_given == "incorrect" else "secondary"
                disabled = feedback_given is not None and feedback_given != "incorrect"
                
                if st.button(
                    get_text('wrong_info'), 
                    key=f"feedback_{chat_id}_incorrect",
                    help="Information seems incorrect",
                    type=button_type,
                    disabled=disabled
                ):
                    if feedback_given is None:
                        send_feedback(chat, "incorrect")
                        st.session_state.feedback_given[chat_id] = "incorrect"
                        st.error(get_text('thanks_review'))
                        st.rerun()
            
            # Show feedback status if given
            if feedback_given:
                feedback_messages = {
                    "helpful": get_text('you_found_helpful'),
                    "not_helpful": get_text('you_marked_not_helpful'),
                    "too_slow": get_text('you_reported_slow'),
                    "incorrect": get_text('you_reported_incorrect')
                }
                st.info(feedback_messages.get(feedback_given, get_text('feedback_recorded')))
            
            # Show sources with quality info
            if chat.get('sources'):
                st.markdown(f"**{get_text('sources')}**")
                
                # Show retrieval quality metrics if available
                stats = chat.get('stats', {})
                if stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(get_text('sources_retrieved'), stats.get('total_retrieved', 0))
                    with col2:
                        st.metric(get_text('avg_similarity'), f"{stats.get('avg_similarity', 0):.1%}")
                    with col3:
                        quality = stats.get('context_quality', 'unknown').title()
                        if lang == 'tr':
                            quality_map = {'High': 'Yüksek', 'Medium': 'Orta', 'Low': 'Düşük', 'Unknown': 'Bilinmiyor'}
                            quality = quality_map.get(quality, quality)
                        st.metric(get_text('context_quality'), quality)
                
                # Show sources
                for doc in chat['sources'][:3]:
                    source = doc.get('source', 'Unknown')
                    similarity = doc.get('similarity_score', 0)
                    airline_info = doc.get('airline', 'Unknown')
                    st.markdown(f"- **{source}** ({similarity:.1%} match) - {airline_info}")

def display_sidebar():
    """Language-aware sidebar"""
    lang = st.session_state.get('language', 'en')
    
    with st.sidebar:
        st.markdown("### 🔗 Connection Status" if lang == 'en' else "### 🔗 Bağlantı Durumu")
        
        api_connected = display_api_status()
        
        if st.button(get_text('reconnect'), use_container_width=True):
            st.session_state.api_connection = None
            st.rerun()
        
        if api_connected:
            # AI Provider selection
            st.markdown(f"### 🤖 {get_text('choose_provider').replace(':', '')}")
            
            provider = st.selectbox(
                get_text('choose_provider'),
                ["OpenAI", "Claude"],
                index=0 if st.session_state.selected_provider == "OpenAI" else 1
            )
            st.session_state.selected_provider = provider
            
            if provider == "Claude":
                models = ['claude-3-haiku-20240307', 'claude-3-5-haiku-20241022', 'claude-sonnet-4-20250514']
            else:
                models = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4']
            
            model = st.selectbox(
                get_text('choose_model'),
                models,
                index=0 if st.session_state.selected_model not in models else models.index(st.session_state.selected_model)
            )
            st.session_state.selected_model = model
            
            # Simple Stats Section
            st.markdown(f"### {get_text('session_stats')}")
            
            if st.session_state.chat_history:
                # Simple feedback stats
                helpful_count = len([fid for fid in st.session_state.feedback_given.values() if fid == "helpful"])
                total_feedback = len(st.session_state.feedback_given)
                satisfaction_rate = (helpful_count / total_feedback * 100) if total_feedback > 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(get_text('total_queries'), len(st.session_state.chat_history))
                with col2:
                    satisfaction_display = f"{satisfaction_rate:.0f}%" if total_feedback > 0 else ("N/A" if lang == 'en' else "Yok")
                    st.metric(get_text('satisfaction'), satisfaction_display)
        
        # Quick Actions
        st.markdown(f"### ⚡ {'Actions' if lang == 'en' else 'Hızlı İşlemler'}")
        
        if st.button(get_text('clear_history'), use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.feedback_given = {}
            st.session_state.session_tracking = {}
            st.rerun()

def display_quick_questions():
    """Language-aware quick question cards"""
    lang = st.session_state.get('language', 'en')
    st.markdown(f"### {get_text('popular_questions')}")
    
    question_categories = QUICK_QUESTIONS.get(lang, QUICK_QUESTIONS['en'])
    
    for category, questions in question_categories.items():
        with st.expander(f"{category}", expanded=False):
            for i, q in enumerate(questions):
                if st.button(
                    f"**{q['title']}**\n{q['desc']}", 
                    key=f"cat_{lang}_{category}_{i}",
                    use_container_width=True,
                    help=f"Ask about: {q['title']}"
                ):
                    st.session_state.current_question = q['title']
                    st.rerun()

def main():
    """Main multilingual application"""
    # Initialize session state FIRST
    init_session_state()
    
    # Display language selector at the top
    display_language_selector()
    
    display_hero_header()
    
    # Check API connection
    if st.session_state.api_connection is None:
        st.session_state.api_connection = find_working_api()
    
    if not st.session_state.api_connection["success"]:
        lang = st.session_state.get('language', 'en')
        error_msg = "🚨 API service required for policy analysis" if lang == 'en' else "🚨 API servisi politika analizi için gereklidir"
        info_msg = "Please ensure the FastAPI service is running" if lang == 'en' else "Lütfen FastAPI servisinin çalıştığından emin olun"
        
        st.error(error_msg)
        st.info(info_msg)
        return
    else:
        st.session_state.api_url = st.session_state.api_connection["url"]
    
    # Sidebar
    display_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Airline selection
        display_airline_selection()
        
        # Question input
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
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "timestamp": datetime.now(),
                    "model": result["model"],
                    "provider": result["provider"],
                    "sources": result["sources"],
                    "airline_filter": st.session_state.selected_airline,
                    "airline_preference": result.get("airline_preference"),
                    "stats": result.get("stats", {}),
                    "preference_stats": result.get("preference_stats", {}),
                    "session_id": result.get("session_id"),
                    "performance": result.get("performance", {}),
                    "language": result.get("language", st.session_state.get('language', 'en'))
                })
                
                st.success(get_text('analysis_complete'))
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
        
        # Chat history
        display_chat_history()
    
    with col2:
        # Quick questions
        display_quick_questions()

if __name__ == "__main__":
    main()