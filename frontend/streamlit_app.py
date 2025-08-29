import streamlit as st
import requests
import time
from datetime import datetime
import os
import json
import random

# Page configuration
st.set_page_config(
    page_title="Multi-Airline Policy Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for Multi-Airline Theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #c41e3a 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .policy-category {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .policy-category:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0, 123, 255, 0.2);
    }
    
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .airline-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 0.2rem;
        text-transform: uppercase;
    }
    
    .badge-turkish {
        background: linear-gradient(45deg, #c41e3a, #e53e3e);
        color: white;
    }
    
    .badge-pegasus {
        background: linear-gradient(45deg, #ff6600, #ff8533);
        color: white;
    }
    
    .badge-general {
        background: linear-gradient(45deg, #6c757d, #8d949e);
        color: white;
    }
    
    .policy-stats {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #2196f3;
    }
    
    .airline-filter-container {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #ff9800;
        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);
    }
    
    .ai-provider-container {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #4caf50;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
    }
    
    .policy-example {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .policy-example:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(156, 39, 176, 0.2);
    }
    
    .api-status {
        padding: 0.5rem;
        border-radius: 8px;
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
    
    .claude-badge {
        background: linear-gradient(45deg, #ff6b35, #ff8c42);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .openai-badge {
        background: linear-gradient(45deg, #00a67e, #00bf88);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 0.2rem;
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

def test_api_connection(api_url, timeout=3):
    """Test API connection with reduced timeout for faster UI response"""
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
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status": "connection_error",
            "error": "API service unavailable",
            "url": api_url
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status": "timeout",
            "error": f"Slow response (>{timeout}s)",
            "url": api_url
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unknown_error",
            "error": str(e)[:50],
            "url": api_url
        }

def find_working_api():
    """Find first working API URL"""
    urls = get_api_urls()
    
    for url in urls:
        result = test_api_connection(url, timeout=3)
        if result["success"]:
            return result
    
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
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'gpt-3.5-turbo'
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = 'OpenAI'
    if 'selected_airline' not in st.session_state:
        st.session_state.selected_airline = 'All Airlines'
    if 'policy_focus' not in st.session_state:
        st.session_state.policy_focus = 'All Policies'

# API Connection Status Widget - OPTIMIZED
def display_api_status():
    """Display API connection status with model preloading information"""
    if st.session_state.api_connection is None:
        st.session_state.api_connection = find_working_api()
    
    connection = st.session_state.api_connection
    
    if connection["success"]:
        st.session_state.api_url = connection["url"]
        
        # Check if models are preloaded
        models_ready = connection.get("models_ready", False)
        
        if models_ready:
            status_icon = "⚡"
            status_text = "Connected (Models Preloaded)"
            status_class = "status-connected"
        else:
            status_icon = "⏳"
            status_text = "Connected (Models Loading...)"
            status_class = "status-connected"
        
        st.markdown(f"""
        <div class="api-status {status_class}">
            {status_icon} API: {connection["url"]}<br>
            Status: {status_text}
        </div>
        """, unsafe_allow_html=True)
        return True
    else:
        st.markdown(f"""
        <div class="api-status status-error">
            ❌ API Connection Failed<br>
            Error: {connection["error"]}<br>
            URL: {connection.get("url", "unknown")}
        </div>
        """, unsafe_allow_html=True)
        return False

# Main header for Multi-Airline Policy Assistant
def display_header():
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Multi-Airline Policy Assistant</h1>
        <h2>🌍 Global Aviation Policy Intelligence</h2>
        <p>Ask questions about airline policies, procedures, and regulations from multiple carriers worldwide</p>
        <small>🔍 Powered by AI • 📚 Multi-source Knowledge Base • 🚀 Real-time Insights</small>
    </div>
    """, unsafe_allow_html=True)

# Enhanced airline badge helper
def get_airline_badge(source):
    """Generate airline badge HTML based on source"""
    source_lower = source.lower()
    
    # Turkish Airlines sources
    turkish_sources = ['checked_baggage', 'carry_on_baggage', 'sports_equipment', 'musical_instruments', 'pets', 'excess_baggage', 'restrictions']
    
    # Pegasus Airlines sources  
    pegasus_sources = ['general_rules', 'baggage_allowance', 'travelling_with_pets', 'extra_services_pricing']
    
    if any(keyword in source_lower for keyword in turkish_sources) or 'turkish' in source_lower:
        return '<span class="airline-badge badge-turkish">🇹🇷 Turkish Airlines</span>'
    elif any(keyword in source_lower for keyword in pegasus_sources) or 'pegasus' in source_lower:
        return '<span class="airline-badge badge-pegasus">🔥 Pegasus Airlines</span>'
    else:
        return '<span class="airline-badge badge-general">✈️ General Policy</span>'

def get_ai_provider_badge(provider):
    """Generate AI provider badge HTML"""
    if provider.lower() == "claude":
        return '<span class="claude-badge">🤖 Claude</span>'
    else:
        return '<span class="openai-badge">🧠 OpenAI</span>'

# Enhanced policy categories and examples
def display_policy_categories():
    st.subheader("📋 Policy Categories")
    
    categories = {
        "🧳 Baggage & Cargo": [
            "What are the baggage weight limits for international flights?",
            "Compare excess baggage fees between Turkish Airlines and Pegasus",
            "What items are prohibited in carry-on baggage?",
            "How to pack sports equipment for air travel?"
        ],
        "🐕 Pet Travel": [
            "Can I travel with my pet in the cabin?",
            "What documents are needed for international pet travel?",
            "Compare pet travel policies between airlines",
            "What are the pet carrier size requirements?"
        ],
        "🎵 Special Items": [
            "How to transport musical instruments on flights?",
            "What are the rules for traveling with sports equipment?",
            "Can I bring electronic devices in checked baggage?",
            "Special handling procedures for fragile items"
        ],
        "✈️ Flight Policies": [
            "What are the check-in time requirements?",
            "Airline policies for flight delays and cancellations",
            "Passenger rights and compensation policies",
            "Seat selection and upgrade policies"
        ],
        "🌍 International Travel": [
            "Visa and document requirements for international flights",
            "Customs and immigration policies",
            "International transit procedures",
            "COVID-19 travel restrictions and requirements"
        ],
        "♿ Accessibility": [
            "Special assistance services for disabled passengers",
            "Wheelchair and mobility aid transport policies",
            "Service animal travel regulations",
            "Medical equipment transport guidelines"
        ]
    }
    
    cols = st.columns(2)
    for i, (category, examples) in enumerate(categories.items()):
        with cols[i % 2]:
            with st.expander(f"{category}", expanded=False):
                st.markdown(f"**{category.split(' ', 1)[1]}**")
                for example in examples:
                    if st.button(f"💡 {example}", key=f"cat_{category}_{examples.index(example)}", use_container_width=True):
                        st.session_state.current_question = example
                        st.rerun()

# Enhanced question handling with CLAUDE SUPPORT - PERFORMANCE OPTIMIZED
def handle_question(question, api_url, selected_model="gpt-3.5-turbo", selected_provider="OpenAI", airline_filter="All Airlines", policy_focus="All Policies"):
    """Enhanced question handling - NOW WITH CLAUDE SUPPORT"""
    
    if not api_url:
        return {
            "success": False,
            "error": "No API connection available"
        }
    
    # OPTIMIZATION 1: Simplified context building
    context_prefix = ""
    if "Pegasus" in airline_filter:
        context_prefix = "Pegasus Airlines: "
    elif "Turkish" in airline_filter:
        context_prefix = "Turkish Airlines: "
    
    enhanced_question = context_prefix + question
    
    # FIXED: Select correct API endpoint based on provider
    if selected_provider == "Claude":
        api_endpoint = f"{api_url}/chat/claude"
        spinner_text = f"Claude analyzing policies from {airline_filter}..."
    else:  # OpenAI (default)
        api_endpoint = f"{api_url}/chat/openai"
        spinner_text = f"OpenAI analyzing policies from {airline_filter}..."
    
    with st.spinner(spinner_text):
        try:
            params = {
                "question": enhanced_question,
                "max_results": 3,
                "similarity_threshold": 0.4,
                "model": selected_model
            }
            
            # Use GET method with correct endpoint
            response = requests.get(
                api_endpoint,
                params=params,
                timeout=25
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    retrieved_docs = data.get("retrieved_docs", [])
                    
                    # Client-side filtering (same as before)
                    if "Pegasus" in airline_filter and retrieved_docs:
                        pegasus_docs = [doc for doc in retrieved_docs[:3]
                                      if 'pegasus' in doc.get("source", "").lower() 
                                         or any(kw in doc.get("source", "").lower() 
                                               for kw in ['general_rules', 'baggage_allowance', 'travelling_with_pets'])]
                        if pegasus_docs:
                            retrieved_docs = pegasus_docs
                    
                    elif "Turkish" in airline_filter and retrieved_docs:
                        turkish_docs = [doc for doc in retrieved_docs[:3]
                                      if any(kw in doc.get("source", "").lower() 
                                           for kw in ['checked_baggage', 'carry_on_baggage', 'sports_equipment', 'pets'])]
                        if turkish_docs:
                            retrieved_docs = turkish_docs
                    
                    return {
                        "success": True,
                        "answer": data["answer"],
                        "sources": len(retrieved_docs),
                        "model": data.get("model_used", selected_model),
                        "provider": selected_provider,
                        "ai_provider_full": data.get("ai_provider", selected_provider),
                        "retrieval_stats": data.get("retrieval_stats", {}),
                        "context_quality": data.get("retrieval_stats", {}).get("context_quality", "good"),
                        "retrieved_docs": retrieved_docs,
                        "filtered": airline_filter != "All Airlines",
                        "policy_coverage": len(set(doc.get("source", "") for doc in retrieved_docs)),
                        "preloaded": data.get("preloaded_model", True)
                    }
                else:
                    return {
                        "success": False,
                        "error": data.get("error", "API processing error"),
                        "details": data
                    }
            else:
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "details": response.text[:200] if len(response.text) > 200 else response.text
                }
                
        except requests.exceptions.ConnectionError:
            st.session_state.api_connection = None
            return {
                "success": False,
                "error": "Connection lost. Click 'Reconnect API' in sidebar."
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Request timed out (25s). {selected_provider} may be processing..."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

# Enhanced chat history with provider metadata - OPTIMIZED
def display_chat_history():
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="policy-stats">
            <h4>👋 Welcome to Multi-Airline Policy Assistant!</h4>
            <p>Ask questions about airline policies - now supports both OpenAI and Claude!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.subheader("💬 Recent Policy Analyses")
    
    # Show last 2 conversations
    for i, chat in enumerate(reversed(st.session_state.chat_history[-2:])):
        # Get provider info for title
        provider = chat.get('provider', 'AI')
        provider_emoji = "🤖" if provider == "Claude" else "🧠"
        
        with st.expander(f"Q ({provider_emoji} {provider}): {chat['question'][:50]}...", expanded=(i == 0)):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f'<div class="chat-message">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            # Source display
            if chat.get('retrieved_docs'):
                st.markdown("**Sources:**")
                for doc in chat['retrieved_docs'][:3]:
                    source = doc.get('source', 'unknown')
                    similarity = doc.get('similarity_score', 0)
                    badge_html = get_airline_badge(source)
                    
                    st.markdown(
                        f"{badge_html} **{source}** ({similarity:.1%})", 
                        unsafe_allow_html=True
                    )
            
            # Enhanced metadata with provider
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                provider_full = chat.get('ai_provider_full', chat.get('provider', 'AI'))
                preloaded = "⚡" if chat.get('preloaded', False) else "⏳"
                st.caption(f"{preloaded} {provider_full}")
            with col2:
                st.caption(f"📚 {chat.get('sources', 0)} sources")
            with col3:
                model_short = chat.get('model', 'unknown')
                if 'claude' in model_short.lower():
                    model_short = model_short.replace('claude-3-5-sonnet', 'C3.5-S').replace('claude-3-sonnet', 'C3-S').replace('claude-3-haiku', 'C3-H')
                elif 'gpt' in model_short.lower():
                    model_short = model_short.replace('gpt-3.5-turbo', 'GPT3.5').replace('gpt-4o-mini', 'GPT4o-M').replace('gpt-4', 'GPT4')
                st.caption(f"🔧 {model_short}")
            with col4:
                st.caption(f"⏱️ {chat['timestamp'].strftime('%H:%M')}")

# Add performance monitoring
def display_performance_status():
    """Display performance monitoring in sidebar"""
    if st.session_state.chat_history:
        recent_chats = st.session_state.chat_history[-5:]  # Last 5 chats
        avg_sources = sum(chat.get('sources', 0) for chat in recent_chats) / len(recent_chats)
        preloaded_count = sum(1 for chat in recent_chats if chat.get('preloaded', False))
        
        # Provider usage stats
        openai_count = sum(1 for chat in recent_chats if chat.get('provider') == 'OpenAI')
        claude_count = sum(1 for chat in recent_chats if chat.get('provider') == 'Claude')
        
        st.subheader("⚡ Performance")
        st.metric("Avg Sources", f"{avg_sources:.1f}")
        st.metric("Preloaded Responses", f"{preloaded_count}/{len(recent_chats)}")
        
        # Provider usage
        if openai_count > 0:
            st.text(f"🧠 OpenAI: {openai_count}")
        if claude_count > 0:
            st.text(f"🤖 Claude: {claude_count}")
        
        if preloaded_count == len(recent_chats):
            st.success("🚀 All models preloaded!")
        elif preloaded_count > 0:
            st.warning(f"⚡ {preloaded_count}/{len(recent_chats)} preloaded")
        else:
            st.error("⏳ Models not preloaded")

# Enhanced sidebar with CLAUDE SUPPORT
def display_sidebar():
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # API Connection Section
        st.subheader("🔗 API Connection")
        api_connected = display_api_status()
        
        if st.button("🔄 Reconnect API"):
            with st.spinner("Reconnecting..."):
                st.session_state.api_connection = find_working_api()
                st.rerun()
        
        if api_connected:
            # Performance monitoring
            display_performance_status()
            
            st.markdown("---")
            st.subheader("🤖 AI PROVIDER")
            
            # AI Provider Selection
            provider_option = st.selectbox(
                "AI Model Provider:",
                [
                    "OpenAI",
                    "Claude"
                ],
                index=0 if st.session_state.get('selected_provider', 'OpenAI') == 'OpenAI' else 1,
                help="Choose between OpenAI GPT models and Claude models"
            )
            
            # Dynamic model selection with CORRECT model names
            if provider_option == "Claude":
                available_models = [
                    "claude-3-haiku-20240307",      # Confirmed working
                    "claude-3-5-haiku-20241022",    # Fastest new model
                    "claude-3-7-sonnet-20250219",   # High performance with extended thinking
                    "claude-sonnet-4-20250514",     # High performance model
                    "claude-opus-4-20250514",       # Previous flagship  
                    "claude-opus-4-1-20250805"      # Most capable
                ]
                
                # Model descriptions for better UX
                model_descriptions = {
                    "claude-3-haiku-20240307": "Claude Haiku 3 (Confirmed Working)",
                    "claude-3-5-haiku-20241022": "Claude Haiku 3.5 (Fastest)",
                    "claude-3-7-sonnet-20250219": "Claude Sonnet 3.7 (Extended Thinking)",
                    "claude-sonnet-4-20250514": "Claude Sonnet 4 (High Performance)",
                    "claude-opus-4-20250514": "Claude Opus 4 (Previous Flagship)",
                    "claude-opus-4-1-20250805": "Claude Opus 4.1 (Most Capable)"
                }
                
                default_model = "claude-3-haiku-20240307"  # Use confirmed working model
                model_help = """Model Tiers:
• Haiku: Fastest & most cost-effective
• Sonnet: Balanced performance & intelligence  
• Opus: Highest intelligence & capability
• 4.x series: Latest with extended thinking"""
                
            else:  # OpenAI
                available_models = [
                    "gpt-3.5-turbo", 
                    "gpt-4o-mini", 
                    "gpt-4"
                ]
                model_descriptions = {
                    "gpt-3.5-turbo": "GPT-3.5 Turbo (Fastest)",
                    "gpt-4o-mini": "GPT-4o Mini (Balanced)",
                    "gpt-4": "GPT-4 (Most Comprehensive)"
                }
                default_model = "gpt-3.5-turbo"
                model_help = "gpt-3.5-turbo: Fastest\ngpt-4o-mini: Balanced\ngpt-4: Most comprehensive"
            
            # Model selection with descriptions
            selected_model_key = st.selectbox(
                "Model:",
                available_models,
                format_func=lambda x: model_descriptions.get(x, x),
                index=0 if st.session_state.get('selected_model', default_model) not in available_models 
                      else available_models.index(st.session_state.get('selected_model', default_model)),
                help=model_help
            )
            
            # Store actual model API name
            selected_model = selected_model_key
            
            # Store selections
            st.session_state.selected_model = selected_model
            st.session_state.selected_provider = provider_option
            
            # Display current AI selection with better formatting
            if provider_option == "Claude":
                model_display = model_descriptions.get(selected_model, selected_model)
                st.success(f"🤖 **CLAUDE**: {model_display}")
                
                # Show status for different models
                if selected_model == "claude-3-haiku-20240307":
                    st.info("✅ Confirmed working model")
                elif selected_model == "claude-3-5-haiku-20241022":
                    st.info("⚡ Fastest new model - test recommended")
                elif "sonnet" in selected_model:
                    st.warning("🔬 High-performance model - verify API access")
                elif "opus" in selected_model:
                    st.warning("👑 Premium model - requires higher tier access")
                    
            else:
                model_display = model_descriptions.get(selected_model, selected_model)  
                st.success(f"🧠 **OPENAI**: {model_display}")
            
            st.markdown("---")
            st.subheader("✈️ AIRLINE FILTER")
            
            airline_option = st.selectbox(
                "🏢 Select Airlines:",
                [
                    "🌍 All Airlines",
                    "🇹🇷 Turkish Airlines Only", 
                    "🔥 Pegasus Airlines Only"
                ],
                index=0,
                help="Filter analysis to specific airline policies"
            )
            
            # Process airline selection
            if "Turkish Airlines Only" in airline_option:
                st.session_state.selected_airline = "Turkish Airlines Only"
            elif "Pegasus Airlines Only" in airline_option:
                st.session_state.selected_airline = "Pegasus Airlines Only"
            else:
                st.session_state.selected_airline = "All Airlines"
            
            # Policy Focus Filter
            st.markdown("---")
            st.subheader("📋 POLICY FOCUS")
            
            policy_focus = st.selectbox(
                "🎯 Focus Area:",
                [
                    "All Policies",
                    "Baggage & Cargo",
                    "Pet Travel",
                    "Special Items",
                    "Flight Operations", 
                    "International Travel",
                    "Accessibility Services"
                ],
                help="Focus analysis on specific policy areas"
            )
            st.session_state.policy_focus = policy_focus
            
            # Display current filters
            if st.session_state.selected_airline != "All Airlines":
                if "Pegasus" in st.session_state.selected_airline:
                    st.success("🔥 **PEGASUS FOCUS**")
                elif "Turkish" in st.session_state.selected_airline:
                    st.success("🇹🇷 **TURKISH FOCUS**")
            
            if policy_focus != "All Policies":
                st.info(f"🎯 **{policy_focus.upper()} FOCUS**")
            
        else:
            # Defaults when API not connected
            selected_model = "gpt-3.5-turbo"
            provider_option = "OpenAI"
            st.warning("Connect to API to access all features")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.success("History cleared!")
            st.rerun()
        
        return selected_model, provider_option, st.session_state.selected_airline, st.session_state.policy_focus

# Main application
def main():
    init_session_state()
    display_header()
    
    # Get settings from sidebar - NOW INCLUDING PROVIDER
    model, provider, airline_filter, policy_focus = display_sidebar()
    
    # Check API connection
    api_connected = display_api_status()
    
    if not api_connected:
        st.error("🚨 **API Connection Required**")
        st.info("Please ensure the FastAPI service is running for policy analysis.")
        with st.expander("🛠️ Troubleshooting"):
            st.markdown("""
            **Docker Setup:**
            - Check containers: `docker-compose ps`
            - View API logs: `docker-compose logs api`
            - Restart services: `docker-compose restart`
            
            **Local Setup:**
            - Start API: `uvicorn myapp:app --host 0.0.0.0 --port 8000`
            - Test health: `curl http://localhost:8000/health`
            """)
        return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # AI Provider Display
        provider_icon = "🤖" if provider == "Claude" else "🧠"
        model_display = model
        
        # Simplify model display names
        if provider == "Claude":
            if "claude-3-5-sonnet" in model:
                model_display = "Claude 3.5 Sonnet"
            elif "claude-3-sonnet" in model:
                model_display = "Claude 3 Sonnet"
            elif "claude-3-haiku" in model:
                model_display = "Claude 3 Haiku"
        else:
            if model == "gpt-3.5-turbo":
                model_display = "GPT-3.5 Turbo"
            elif model == "gpt-4o-mini":
                model_display = "GPT-4o Mini"
            elif model == "gpt-4":
                model_display = "GPT-4"
        
        st.markdown(f"""
        <div class="ai-provider-container">
            <h3>{provider_icon} {provider}: {model_display}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Airline Filter buttons
        st.markdown("""
        <div class="airline-filter-container">
            <h3>✈️ Quick Airline Selection</h3>
        """, unsafe_allow_html=True)
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            if st.button("🌍 All Airlines", use_container_width=True, 
                        type="primary" if st.session_state.get('selected_airline', '') == 'All Airlines' else "secondary"):
                st.session_state.selected_airline = "All Airlines"
                st.rerun()
                
        with filter_col2:
            if st.button("🇹🇷 Turkish Airlines", use_container_width=True,
                        type="primary" if "Turkish" in st.session_state.get('selected_airline', '') else "secondary"):
                st.session_state.selected_airline = "Turkish Airlines Only"
                st.rerun()
                
        with filter_col3:
            if st.button("🔥 Pegasus Airlines", use_container_width=True,
                        type="primary" if "Pegasus" in st.session_state.get('selected_airline', '') else "secondary"):
                st.session_state.selected_airline = "Pegasus Airlines Only"
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show active filters
        current_filter = st.session_state.get('selected_airline', 'All Airlines')
        current_policy = st.session_state.get('policy_focus', 'All Policies')
        
        if current_filter != "All Airlines" or current_policy != "All Policies":
            filter_text = f"🎯 **Active Filters:** {current_filter}"
            if current_policy != "All Policies":
                filter_text += f" | {current_policy}"
            st.success(filter_text)
        
        st.subheader("💬 Ask About Airline Policies")
        
        # Question input
        question = st.text_area(
            f"What would you like to know about airline policies? (Powered by {provider})",
            value=st.session_state.get('current_question', ''),
            placeholder=f"Example: Compare baggage policies between Turkish Airlines and Pegasus Airlines (will be analyzed by {provider})",
            height=100,
            key="question_input"
        )
        
        # Action buttons
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            ask_clicked = st.button(f"🚀 Analyze with {provider}", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🎲 Random Question", use_container_width=True):
                examples = [
                    "Compare excess baggage fees between Turkish Airlines and Pegasus",
                    "What are the pet travel requirements for international flights?",
                    "Which airline offers better compensation for flight delays?",
                    "How do airlines handle musical instrument transport?",
                    "What COVID-19 policies do Turkish airlines have?",
                    "Compare frequent flyer programs benefits",
                    "Special assistance services for disabled passengers",
                    "International transit policies and procedures"
                ]
                st.session_state.current_question = random.choice(examples)
                st.rerun()
        
        # Handle question - NOW WITH PROVIDER SUPPORT
        if ask_clicked and question.strip():
            result = handle_question(
                question, 
                st.session_state.api_url,
                model,  # selected model
                provider,  # selected provider
                st.session_state.selected_airline,
                st.session_state.policy_focus
            )
            
            if result["success"]:
                # Add to history with provider info
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "timestamp": datetime.now(),
                    "model": result["model"],
                    "provider": provider,
                    "ai_provider_full": result.get("ai_provider_full", provider),
                    "sources": result["sources"],
                    "context_quality": result.get("context_quality", "unknown"),
                    "retrieved_docs": result.get("retrieved_docs", []),
                    "filtered": result.get("filtered", False),
                    "airline_filter": st.session_state.selected_airline,
                    "policy_focus": st.session_state.policy_focus,
                    "policy_coverage": result.get("policy_coverage", 0),
                    "preloaded": result.get("preloaded", True)
                })
                
                st.success(f"✅ Policy analysis complete with {provider}!")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
        
        # Display chat history
        display_chat_history()
    
    with col2:
        # Policy categories
        display_policy_categories()

if __name__ == "__main__":
    main()