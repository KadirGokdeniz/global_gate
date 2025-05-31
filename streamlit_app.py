import streamlit as st
import requests
import time
from datetime import datetime
import os
import json

# Page configuration
st.set_page_config(
    page_title="Turkish Airlines - Baggage Assistant",
    page_icon="🇹🇷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #c41e3a 0%, #8b1538 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        border-left: 5px solid #c41e3a;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .example-button {
        background: #f8f9fa;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }
    
    .status-connected {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced API Configuration
def get_api_urls():
    """Get multiple API URL options"""
    urls = [
        os.getenv('DEFAULT_API_URL', 'http://api:8000'),  # Container network
        'http://localhost:8000',  # Local development
        'http://127.0.0.1:8000',  # Alternative local
    ]
    return urls

def test_api_connection(api_url, timeout=5):
    """Test API connection with detailed error info"""
    try:
        response = requests.get(f"{api_url}/health", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "status": "connected",
                "data": data,
                "url": api_url
            }
        else:
            return {
                "success": False,
                "status": f"http_error_{response.status_code}",
                "error": f"HTTP {response.status_code}",
                "url": api_url
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status": "connection_error",
            "error": "Connection refused - API service might be down",
            "url": api_url
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status": "timeout",
            "error": f"Timeout after {timeout}s",
            "url": api_url
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unknown_error",
            "error": str(e),
            "url": api_url
        }

def find_working_api():
    """Find first working API URL"""
    urls = get_api_urls()
    
    for url in urls:
        result = test_api_connection(url, timeout=3)
        if result["success"]:
            return result
    
    # No working API found
    return {
        "success": False,
        "status": "all_failed",
        "error": "No working API found",
        "tested_urls": urls
    }

# Initialize session state
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_connection' not in st.session_state:
        st.session_state.api_connection = None
    if 'api_url' not in st.session_state:
        st.session_state.api_url = None

# API Connection Status Widget
def display_api_status():
    """Display API connection status"""
    if st.session_state.api_connection is None:
        st.session_state.api_connection = find_working_api()
    
    connection = st.session_state.api_connection
    
    if connection["success"]:
        st.session_state.api_url = connection["url"]
        st.markdown(f"""
        <div class="api-status status-connected">
            ✅ API Connected: {connection["url"]}<br>
            Status: {connection.get("data", {}).get("status", "healthy")}
        </div>
        """, unsafe_allow_html=True)
        return True
    else:
        st.markdown(f"""
        <div class="api-status status-error">
            ❌ API Connection Failed<br>
            Error: {connection["error"]}<br>
            Tested URLs: {', '.join(connection.get("tested_urls", []))}
        </div>
        """, unsafe_allow_html=True)
        return False

# Main header
def display_header():
    st.markdown("""
    <div class="main-header">
        <h1>🇹🇷 Turkish Airlines</h1>
        <h2>Baggage Policy Assistant</h2>
        <p>Ask questions about baggage policies and get instant AI-powered answers</p>
    </div>
    """, unsafe_allow_html=True)

# Example questions
def display_examples():
    st.subheader("💡 Popular Questions")
    
    examples = [
        "What is the baggage weight limit for economy class?",
        "Can I bring sports equipment on Turkish Airlines?",
        "How much do excess baggage fees cost?",
        "What are the carry-on baggage size restrictions?",
        "Can I travel with my pet on Turkish Airlines?",
        "What musical instruments can I bring on board?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💼 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.current_question = example
                st.rerun()

# Enhanced question handling
def handle_question(question, api_url):
    """Enhanced question handling with better error reporting"""
    
    if not api_url:
        return {
            "success": False,
            "error": "No API connection available"
        }
    
    with st.spinner("🤖 Getting your answer from Turkish Airlines AI..."):
        try:
            # Debug: Show what we're sending
            with st.expander("🔍 Debug Info", expanded=False):
                st.json({
                    "api_url": api_url,
                    "endpoint": "/chat/openai",
                    "question": question
                })
            
            response = requests.post(
                f"{api_url}/chat/openai",
                params={
                    "question": question,
                    "max_results": 3,
                    "similarity_threshold": 0.5,
                    "model": "gpt-3.5-turbo"
                },
                timeout=45  # Increased timeout for RAG processing
            )
            
            # Debug: Show response details
            with st.expander("🔍 API Response Debug", expanded=False):
                st.text(f"Status Code: {response.status_code}")
                try:
                    response_json = response.json()
                    st.json(response_json)
                except:
                    st.text(f"Raw Response: {response.text[:500]}...")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return {
                        "success": True,
                        "answer": data["answer"],
                        "sources": len(data.get("retrieved_docs", [])),
                        "model": data.get("model_used", "gpt-3.5-turbo"),
                        "retrieval_stats": data.get("retrieval_stats", {}),
                        "context_quality": data.get("retrieval_stats", {}).get("context_quality", "unknown")
                    }
                else:
                    return {
                        "success": False,
                        "error": data.get("error", "Unknown API error"),
                        "details": data
                    }
            else:
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "details": response.text[:200]
                }
                
        except requests.exceptions.ConnectionError:
            # Try to reconnect
            st.session_state.api_connection = None
            return {
                "success": False,
                "error": "Connection lost. Click 'Reconnect API' in sidebar."
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out. The AI might be processing a complex query."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

# Enhanced chat history
def display_chat_history():
    if not st.session_state.chat_history:
        st.info("👋 Ask your first question about Turkish Airlines baggage policies!")
        return
    
    st.subheader("💬 Recent Conversations")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):  # Show last 3
        with st.expander(f"Q: {chat['question'][:60]}...", expanded=(i == 0)):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f'<div class="chat-message">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            # Enhanced metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption(f"🤖 {chat.get('model', 'AI')}")
            with col2:
                st.caption(f"📚 {chat.get('sources', 0)} sources")
            with col3:
                quality = chat.get('context_quality', 'unknown')
                quality_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(quality, "⚪")
                st.caption(f"{quality_emoji} {quality}")
            with col4:
                st.caption(f"⏰ {chat['timestamp'].strftime('%H:%M')}")

# Enhanced sidebar
def display_sidebar():
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # API Connection Section
        st.subheader("🔗 API Connection")
        
        # Connection status
        api_connected = display_api_status()
        
        # Manual reconnect
        if st.button("🔄 Reconnect API"):
            with st.spinner("Reconnecting..."):
                st.session_state.api_connection = find_working_api()
                st.rerun()
        
        # Manual URL override
        with st.expander("⚙️ Manual API URL"):
            custom_url = st.text_input("API URL:", value=st.session_state.get('api_url', ''))
            if st.button("Test Custom URL") and custom_url:
                result = test_api_connection(custom_url)
                if result["success"]:
                    st.session_state.api_url = custom_url
                    st.success("✅ Custom URL works!")
                else:
                    st.error(f"❌ {result['error']}")
        
        # Model selection (only if connected)
        if api_connected:
            st.subheader("🤖 AI Settings")
            model = st.selectbox(
                "Model:",
                ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"],
                help="gpt-3.5-turbo is faster and cheaper"
            )
        else:
            model = "gpt-3.5-turbo"
            st.warning("Connect to API to change settings")
        
        # Quick actions
        st.subheader("⚡ Actions")
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.success("History cleared!")
            st.rerun()
        
        # Stats
        if st.session_state.chat_history:
            st.subheader("📊 Session Stats")
            st.metric("Questions Asked", len(st.session_state.chat_history))
            
            # Quality distribution
            qualities = [chat.get('context_quality', 'unknown') for chat in st.session_state.chat_history]
            quality_counts = {q: qualities.count(q) for q in set(qualities)}
            for quality, count in quality_counts.items():
                emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(quality, "⚪")
                st.text(f"{emoji} {quality}: {count}")
        
        return model

# Main app
def main():
    init_session_state()
    display_header()
    
    # Check API connection first
    api_connected = display_api_status()
    
    if not api_connected:
        st.error("🚨 **API Connection Required**")
        st.info("Please check that the FastAPI service is running and accessible.")
        with st.expander("🛠️ Troubleshooting"):
            st.markdown("""
            **If running with Docker:**
            - Make sure all containers are up: `docker-compose ps`
            - Check API logs: `docker-compose logs api`
            
            **If running locally:**
            - Start the API: `uvicorn myapp:app --host 0.0.0.0 --port 8000`
            - Check API health: `curl http://localhost:8000/health`
            """)
        return
    
    # Main interface (only if API connected)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Question")
        
        # Question input
        question = st.text_area(
            "What would you like to know about Turkish Airlines baggage policies?",
            value=st.session_state.get('current_question', ''),
            placeholder="Example: What is the baggage weight limit for economy class?",
            height=100,
            key="question_input"
        )
        
        # Ask button
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            ask_clicked = st.button("🚀 Ask Assistant", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🎲 Random Question", use_container_width=True):
                examples = [
                    "What is the baggage weight limit?",
                    "Can I bring sports equipment?", 
                    "How much are excess baggage fees?",
                    "What are carry-on restrictions?",
                    "Can I travel with my pet?",
                    "What about musical instruments?"
                ]
                import random
                st.session_state.current_question = random.choice(examples)
                st.rerun()
        
        # Handle question
        if ask_clicked and question.strip():
            result = handle_question(question, st.session_state.api_url)
            
            if result["success"]:
                # Add to history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "timestamp": datetime.now(),
                    "model": result["model"],
                    "sources": result["sources"],
                    "context_quality": result.get("context_quality", "unknown")
                })
                
                st.success("✅ Answer generated!")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
                if "details" in result:
                    with st.expander("🔍 Error Details"):
                        st.json(result["details"])
        
        # Chat history
        display_chat_history()
    
    with col2:
        display_examples()
    
    # Sidebar
    model = display_sidebar()
    st.session_state.selected_model = model

if __name__ == "__main__":
    main()