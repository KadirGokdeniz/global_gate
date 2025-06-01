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
    initial_sidebar_state="expanded"  # Make sidebar visible by default
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
    if 'selected_airline' not in st.session_state:
        st.session_state.selected_airline = 'All Airlines'
    if 'policy_focus' not in st.session_state:
        st.session_state.policy_focus = 'All Policies'

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

# Popular example questions for multi-airline policies
def display_examples():
    st.subheader("🔥 Popular Questions")
    
    examples = [
        "Compare baggage policies between Turkish Airlines and Pegasus Airlines",
        "What are the pet travel requirements for international flights?",
        "Which airline has better compensation for flight delays?",
        "How to transport musical instruments safely on flights?",
        "What are the COVID-19 travel requirements for different airlines?",
        "Special assistance services comparison between airlines",
        "Duty-free shopping policies and restrictions",
        "Group booking policies and discounts",
        "Frequent flyer program benefits comparison",
        "Airline policies for unaccompanied minors",
        "What items are prohibited in checked vs carry-on baggage?",
        "How do airlines handle lost or damaged luggage?"
    ]
    
    # Group examples by policy type
    policy_types = {
        "🧳 Baggage & Transport": examples[:4],
        "✈️ Service & Compensation": examples[4:8],
        "🎫 Booking & Programs": examples[8:12]
    }
    
    for policy_type, type_examples in policy_types.items():
        st.markdown(f"**{policy_type}**")
        for example in type_examples:
            if st.button(f"🔍 {example}", key=f"example_{example[:20]}", use_container_width=True):
                st.session_state.current_question = example
                st.rerun()
        st.markdown("---")

# Enhanced question handling with policy-focused context
def handle_question(question, api_url, selected_model="gpt-3.5-turbo", airline_filter="All Airlines", policy_focus="All Policies"):
    """Enhanced question handling for multi-airline policy queries"""
    
    if not api_url:
        return {
            "success": False,
            "error": "No API connection available"
        }
    
    # Enhanced question with airline and policy context
    context_parts = []
    
    if "Pegasus" in airline_filter:
        context_parts.append("Focus specifically on Pegasus Airlines policies and procedures.")
    elif "Turkish" in airline_filter:
        context_parts.append("Focus specifically on Turkish Airlines policies and procedures.")
    else:
        context_parts.append("Compare and analyze policies from multiple airlines when relevant.")
    
    if policy_focus != "All Policies":
        context_parts.append(f"Focus on {policy_focus.lower()} related policies.")
    
    # Build enhanced question
    if context_parts:
        enhanced_question = f"{' '.join(context_parts)} {question}"
    else:
        enhanced_question = question
    
    with st.spinner(f"🤖 Analyzing policies from {airline_filter}..."):
        try:
            # Debug info
            with st.expander("🔍 Query Analysis", expanded=False):
                st.json({
                    "api_url": api_url,
                    "original_question": question,
                    "enhanced_question": enhanced_question,
                    "model": selected_model,
                    "airline_filter": airline_filter,
                    "policy_focus": policy_focus
                })
            
            params = {
                "question": enhanced_question,
                "max_results": 5,  # More results for policy comparison
                "similarity_threshold": 0.3,  # Lower threshold for broader policy coverage
                "model": selected_model
            }
            
            response = requests.post(
                f"{api_url}/chat/openai",
                params=params,
                timeout=50
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    retrieved_docs = data.get("retrieved_docs", [])
                    
                    # Apply client-side filtering based on airline preference
                    if "Pegasus" in airline_filter:
                        pegasus_docs = [doc for doc in retrieved_docs 
                                      if any(keyword in doc.get("source", "").lower() 
                                           for keyword in ['pegasus', 'general_rules', 'baggage_allowance', 'travelling_with_pets', 'extra_services'])]
                        if pegasus_docs:
                            retrieved_docs = pegasus_docs
                        else:
                            st.warning("⚠️ Limited Pegasus-specific data found. Showing general policy results.")
                    
                    elif "Turkish" in airline_filter:
                        turkish_docs = [doc for doc in retrieved_docs 
                                      if any(keyword in doc.get("source", "").lower() 
                                           for keyword in ['checked_baggage', 'carry_on_baggage', 'sports_equipment', 'musical_instruments', 'pets', 'excess_baggage', 'restrictions'])]
                        if turkish_docs:
                            retrieved_docs = turkish_docs
                        else:
                            st.warning("⚠️ Limited Turkish Airlines-specific data found. Showing general policy results.")
                    
                    return {
                        "success": True,
                        "answer": data["answer"],
                        "sources": len(retrieved_docs),
                        "model": data.get("model_used", "gpt-3.5-turbo"),
                        "retrieval_stats": data.get("retrieval_stats", {}),
                        "context_quality": data.get("retrieval_stats", {}).get("context_quality", "unknown"),
                        "retrieved_docs": retrieved_docs,
                        "filtered": airline_filter != "All Airlines",
                        "policy_coverage": len(set(doc.get("source", "") for doc in retrieved_docs))
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
            st.session_state.api_connection = None
            return {
                "success": False,
                "error": "Connection lost. Click 'Reconnect API' in sidebar."
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out. Complex policy analysis may take longer."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

# Enhanced chat history with policy metadata
def display_chat_history():
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="policy-stats">
            <h4>👋 Welcome to Multi-Airline Policy Assistant!</h4>
            <p>Ask questions about:</p>
            <ul>
                <li>🧳 Baggage policies and fees</li>
                <li>🐕 Pet travel requirements</li>
                <li>🎵 Special item transport</li>
                <li>✈️ Flight procedures</li>
                <li>🌍 International travel rules</li>
                <li>♿ Accessibility services</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.subheader("💬 Policy Analysis History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
        with st.expander(f"Q: {chat['question'][:80]}...", expanded=(i == 0)):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f'<div class="chat-message">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            # Enhanced source display with policy metadata
            if chat.get('retrieved_docs'):
                st.markdown("**Policy Sources:**")
                
                # Group sources by airline
                turkish_sources = []
                pegasus_sources = []
                other_sources = []
                
                for doc in chat['retrieved_docs'][:5]:
                    source = doc.get('source', 'unknown')
                    if any(keyword in source.lower() for keyword in ['checked_baggage', 'carry_on_baggage', 'sports_equipment', 'musical_instruments', 'pets', 'excess_baggage', 'restrictions']):
                        turkish_sources.append(doc)
                    elif any(keyword in source.lower() for keyword in ['general_rules', 'baggage_allowance', 'travelling_with_pets', 'extra_services']):
                        pegasus_sources.append(doc)
                    else:
                        other_sources.append(doc)
                
                # Display grouped sources
                source_groups = [
                    ("🇹🇷 Turkish Airlines Sources", turkish_sources),
                    ("🔥 Pegasus Airlines Sources", pegasus_sources),
                    ("✈️ General Sources", other_sources)
                ]
                
                for group_name, group_docs in source_groups:
                    if group_docs:
                        st.markdown(f"**{group_name}:**")
                        for doc in group_docs:
                            similarity = doc.get('similarity_score', 0)
                            badge_html = get_airline_badge(doc.get('source', ''))
                            
                            st.markdown(
                                f"{badge_html} "
                                f"**{doc.get('source', 'Unknown')}** "
                                f"(Match: {similarity:.1%})", 
                                unsafe_allow_html=True
                            )
            
            # Enhanced metadata display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption(f"🤖 {chat.get('model', 'AI')}")
            with col2:
                st.caption(f"📚 {chat.get('sources', 0)} sources")
            with col3:
                coverage = chat.get('policy_coverage', 0)
                st.caption(f"📋 {coverage} policy areas")
            with col4:
                st.caption(f"⏰ {chat['timestamp'].strftime('%H:%M')}")

# Enhanced sidebar with policy focus
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
            
            st.markdown("---")
            st.subheader("🤖 AI Configuration")
            
            model = st.selectbox(
                "Model:",
                ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"],
                index=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"].index(
                    st.session_state.get('selected_model', 'gpt-3.5-turbo')
                ),
                help="gpt-3.5-turbo: Fast & efficient\ngpt-4o-mini: Balanced\ngpt-4: Most comprehensive"
            )
            st.session_state.selected_model = model
            
        else:
            model = "gpt-3.5-turbo"
            st.warning("Connect to API to access filters")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.success("History cleared!")
            st.rerun()
        
        # Enhanced Stats
        if st.session_state.chat_history:
            st.subheader("📊 Session Analytics")
            st.metric("Policy Queries", len(st.session_state.chat_history))
            
            # Policy coverage analysis
            all_sources = []
            for chat in st.session_state.chat_history:
                for doc in chat.get('retrieved_docs', []):
                    all_sources.append(doc.get('source', ''))
            
            unique_sources = len(set(all_sources))
            st.metric("Policy Sources Used", unique_sources)
            
            # Airline usage distribution
            turkish_usage = sum(1 for s in all_sources if any(kw in s.lower() for kw in ['checked_baggage', 'carry_on', 'sports', 'musical', 'excess', 'restrictions']))
            pegasus_usage = sum(1 for s in all_sources if any(kw in s.lower() for kw in ['general_rules', 'baggage_allowance', 'travelling_with_pets', 'extra_services']))
            
            if turkish_usage > 0:
                st.text(f"🇹🇷 Turkish Airlines: {turkish_usage}")
            if pegasus_usage > 0:
                st.text(f"🔥 Pegasus Airlines: {pegasus_usage}")
        
        return model, st.session_state.selected_airline, st.session_state.policy_focus

# Main application
def main():
    init_session_state()
    display_header()
    
    # Get settings from sidebar
    model, airline_filter, policy_focus = display_sidebar()
    
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
        # Airline Filter in Main Area
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
            "What would you like to know about airline policies?",
            value=st.session_state.get('current_question', ''),
            placeholder="Example: Compare baggage policies between Turkish Airlines and Pegasus Airlines",
            height=100,
            key="question_input"
        )
        
        # Action buttons
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            ask_clicked = st.button("🚀 Analyze Policies", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🎲 Random Policy Question", use_container_width=True):
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
        
        # Handle question
        if ask_clicked and question.strip():
            result = handle_question(
                question, 
                st.session_state.api_url,
                st.session_state.selected_model,
                st.session_state.selected_airline,
                st.session_state.policy_focus
            )
            
            if result["success"]:
                # Add to history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "timestamp": datetime.now(),
                    "model": result["model"],
                    "sources": result["sources"],
                    "context_quality": result.get("context_quality", "unknown"),
                    "retrieved_docs": result.get("retrieved_docs", []),
                    "filtered": result.get("filtered", False),
                    "airline_filter": st.session_state.selected_airline,
                    "policy_focus": st.session_state.policy_focus,
                    "policy_coverage": result.get("policy_coverage", 0)
                })
                
                st.success("✅ Policy analysis complete!")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
        
        # Display chat history
        display_chat_history()
    
    with col2:
        # Policy categories and examples
        display_policy_categories()

if __name__ == "__main__":
    main()