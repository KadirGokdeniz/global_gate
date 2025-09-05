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

# Enhanced CSS with beautiful animations and effects
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
    
    /* Floating particles animation in hero */
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
    
    /* Simplified feedback section */
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
    
    /* Sidebar sections */
    .sidebar-section {
        margin: 1rem 0;
        padding: 1rem 0;
        border-bottom: 1px solid #e9ecef;
    }
    
    .sidebar-section h4 {
        color: #495057;
        margin-bottom: 0.5rem;
        font-weight: 600;
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
</style>
""", unsafe_allow_html=True)

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

# Session state initialization
def init_session_state():
    defaults = {
        'chat_history': [],
        'api_connection': None,
        'api_url': None,
        'selected_model': 'gpt-3.5-turbo',
        'selected_provider': 'OpenAI',
        'selected_airline': 'All Airlines',
        'current_question': '',
        'show_advanced': False,
        'feedback_given': {},
        'session_tracking': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_hero_header():
    """Clean hero header"""
    st.markdown("""
    <div class="hero-header">
        <h1>✈️ Airline Policy Assistant</h1>
        <p>Get instant answers about airline policies powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

def display_question_input():
    """Clean question input area"""
    st.markdown("### Ask Your Question")
    
    question = st.text_area(
        "",
        value=st.session_state.get('current_question', ''),
        placeholder="What would you like to know about airline policies?",
        height=100,
        key="question_input",
        label_visibility="collapsed"
    )
    
    

    ask_clicked = st.button(
        f"Ask {st.session_state.selected_provider}", 
        type="primary", 
        use_container_width=True,
        disabled=not question.strip()
    )
    
    return ask_clicked, question

def handle_question_optimized(question, api_url, model, provider, airline_filter):
    """Simplified question handling"""
    if not api_url:
        return {"success": False, "error": "No API connection"}
    
    endpoint = f"{api_url}/chat/claude" if provider == "Claude" else f"{api_url}/chat/openai"
    
    context_prefix = ""
    if "Turkish" in airline_filter:
        context_prefix = "Turkish Airlines: "
    elif "Pegasus" in airline_filter:
        context_prefix = "Pegasus Airlines: "
    
    enhanced_question = context_prefix + question
    
    try:
        with st.spinner(f"🤔 {provider} is analyzing airline policies..."):
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
                    "session_id": data.get("session_id"),
                    "answer": data["answer"],
                    "sources": data.get("sources", []),
                    "model": data.get("model_used", model),
                    "provider": provider,
                    "stats": data.get("stats", {}),
                    "performance": data.get("performance", {})
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
    """Simplified chat history display"""
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-section">
            <h3>Welcome to AI Assistant!</h3>
            <p>Ask your first question about airline policies above. Features:</p>
            <ul>
                <li>✅ Smart Policy Search</li>
                <li>⚡ Fast Response Times</li>
                <li>📊 Quality Tracking</li>
                <li>🎯 User Satisfaction</li>
            </ul>
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
                st.markdown(f"""
                <div class="session-info">
                    Session ID: {session_id[:16]}...
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Simplified Feedback Section
            st.markdown("""
            <div class="feedback-section">
                <h4>📝 Feedback</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if feedback already given for this chat
            feedback_given = st.session_state.feedback_given.get(chat_id, None)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                button_type = "primary" if feedback_given == "helpful" else "secondary"
                disabled = feedback_given is not None and feedback_given != "helpful"
                
                if st.button(
                    "👍 Helpful", 
                    key=f"feedback_{chat_id}_helpful",
                    help="This answer was helpful",
                    type=button_type,
                    disabled=disabled
                ):
                    if feedback_given is None:
                        send_feedback(chat, "helpful")
                        st.session_state.feedback_given[chat_id] = "helpful"
                        st.success("Thanks for your feedback!")
                        st.rerun()
            
            with col2:
                button_type = "primary" if feedback_given == "not_helpful" else "secondary"
                disabled = feedback_given is not None and feedback_given != "not_helpful"
                
                if st.button(
                    "👎 Not Helpful", 
                    key=f"feedback_{chat_id}_not_helpful",
                    help="This answer was not helpful",
                    type=button_type,
                    disabled=disabled
                ):
                    if feedback_given is None:
                        send_feedback(chat, "not_helpful")
                        st.session_state.feedback_given[chat_id] = "not_helpful"
                        st.info("Thanks for your feedback!")
                        st.rerun()
            
            with col3:
                button_type = "primary" if feedback_given == "too_slow" else "secondary"
                disabled = feedback_given is not None and feedback_given != "too_slow"
                
                if st.button(
                    "⏱️ Too Slow", 
                    key=f"feedback_{chat_id}_too_slow",
                    help="Response was too slow",
                    type=button_type,
                    disabled=disabled
                ):
                    if feedback_given is None:
                        send_feedback(chat, "too_slow")
                        st.session_state.feedback_given[chat_id] = "too_slow"
                        st.warning("We'll work on speed!")
                        st.rerun()
            
            with col4:
                button_type = "primary" if feedback_given == "incorrect" else "secondary"
                disabled = feedback_given is not None and feedback_given != "incorrect"
                
                if st.button(
                    "❌ Wrong Info", 
                    key=f"feedback_{chat_id}_incorrect",
                    help="Information seems incorrect",
                    type=button_type,
                    disabled=disabled
                ):
                    if feedback_given is None:
                        send_feedback(chat, "incorrect")
                        st.session_state.feedback_given[chat_id] = "incorrect"
                        st.error("Thanks, we'll review this!")
                        st.rerun()
            
            # Show feedback status if given
            if feedback_given:
                feedback_messages = {
                    "helpful": "✅ You found this helpful",
                    "not_helpful": "⚠️ You marked this as not helpful",
                    "too_slow": "⏱️ You reported this was too slow",
                    "incorrect": "❌ You reported incorrect information"
                }
                st.info(feedback_messages.get(feedback_given, "Feedback recorded"))
            
            # Show sources with quality info
            if chat.get('sources'):
                st.markdown("**Sources:**")
                
                # Show retrieval quality metrics if available
                stats = chat.get('stats', {})
                if stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sources Retrieved", stats.get('total_retrieved', 0))
                    with col2:
                        st.metric("Avg Similarity", f"{stats.get('avg_similarity', 0):.1%}")
                    with col3:
                        st.metric("Context Quality", stats.get('context_quality', 'unknown').title())
                
                # Show sources
                for doc in chat['sources'][:3]:
                    source = doc.get('source', 'Unknown')
                    similarity = doc.get('similarity_score', 0)
                    st.markdown(f"- **{source}** ({similarity:.1%} match)")

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
                🔄 API Connected (Loading...)
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
    """Simplified sidebar"""
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
            # AI Provider selection
            st.markdown("""
            <div class="sidebar-section">
                <h4>🤖 AI Assistant</h4>
            </div>
            """, unsafe_allow_html=True)
            
            provider = st.selectbox(
                "Choose Provider:",
                ["OpenAI", "Claude"],
                index=0 if st.session_state.selected_provider == "OpenAI" else 1
            )
            st.session_state.selected_provider = provider
            
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
            
            # Simple Stats Section
            st.markdown("""
            <div class="sidebar-section">
                <h4>📊 Session Stats</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.chat_history:
                recent = st.session_state.chat_history[-5:]
                
                # Simple feedback stats
                helpful_count = len([fid for fid in st.session_state.feedback_given.values() if fid == "helpful"])
                total_feedback = len(st.session_state.feedback_given)
                satisfaction_rate = (helpful_count / total_feedback * 100) if total_feedback > 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Queries", len(st.session_state.chat_history))
                with col2:
                    st.metric("Satisfaction", f"{satisfaction_rate:.0f}%" if total_feedback > 0 else "N/A")
        
        # Quick Actions
        st.markdown("""
        <div class="sidebar-section">
            <h4>⚡ Actions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.feedback_given = {}
            st.session_state.session_tracking = {}
            st.rerun()

def display_quick_questions():
    """Simplified quick question cards"""
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

def main():
    """Main application"""
    init_session_state()
    display_hero_header()
    
    # Check API connection
    if st.session_state.api_connection is None:
        st.session_state.api_connection = find_working_api()
    
    if not st.session_state.api_connection["success"]:
        st.error("🚨 API service required for policy analysis")
        st.info("Please ensure the FastAPI service is running")
        return
    else:
        st.session_state.api_url = st.session_state.api_connection["url"]
    
    # Sidebar
    display_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Airline selection
        st.markdown("### Choose Airlines")
        
        airline_options = ["All Airlines", "Turkish Airlines Only", "Pegasus Airlines Only"]
        selected_airline = st.selectbox(
            "Select airline focus:",
            airline_options,
            index=airline_options.index(st.session_state.selected_airline) if st.session_state.selected_airline in airline_options else 0,
            key="airline_selectbox"
        )
        
        st.session_state.selected_airline = selected_airline
        
        # Airline feedback
        if selected_airline == "Turkish Airlines Only":
            st.error("🇹🇷 Turkish Airlines Focus - Queries will prioritize Turkish Airlines policies")
        elif selected_airline == "Pegasus Airlines Only":
            st.warning("✈️ Pegasus Airlines Focus - Queries will prioritize Pegasus Airlines policies")  
        else:
            st.success("🌍 All Airlines - Queries will search across all available airline policies")
        
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
                    "stats": result.get("stats", {}),
                    "session_id": result.get("session_id"),
                    "performance": result.get("performance", {})
                })
                
                st.success("✅ Analysis complete!")
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