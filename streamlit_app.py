import streamlit as st
import requests
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Turkish Airlines - Baggage Assistant",
    page_icon="🇹🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Turkish Airlines branding
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
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header h3 {
        margin: 0.5rem 0;
        font-size: 1.5rem;
        opacity: 0.9;
    }
    
    .main-header p {
        margin: 0;
        font-size: 1.1rem;
        opacity: 0.8;
    }
    
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        border-left: 5px solid #c41e3a;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .source-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1976d2;
        font-size: 0.9rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .example-question {
        background: #f0f0f0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        border-left: 3px solid #c41e3a;
    }
    
    .example-question:hover {
        background: #e0e0e0;
        transform: translateX(5px);
    }
    
    .cost-tracker {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0.0
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    if 'api_status' not in st.session_state:
        st.session_state.api_status = "unknown"

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🇹🇷 Turkish Airlines</h1>
        <h3>AI-Powered Baggage Policy Assistant</h3>
        <p>Get instant, accurate answers about baggage policies using advanced AI technology</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display and handle sidebar configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Settings
        api_url = st.text_input(
            "API Base URL", 
            value="http://localhost:8000",
            help="FastAPI server URL"
        )
        
        # OpenAI Model Selection  
        model_choice = st.selectbox(
            "🤖 OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="gpt-3.5-turbo: Fast & cheap, gpt-4: More accurate but expensive"
        )
        
        # Search Settings
        st.subheader("🔍 Search Settings")
        max_results = st.slider(
            "Max Documents Retrieved", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="More documents = better context but higher cost"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.3, 
            step=0.1,
            help="Lower = more results, Higher = more relevant"
        )
        
        # Usage Statistics
        st.subheader("📊 Usage Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Queries", 
                st.session_state.total_queries,
                help="Number of questions asked"
            )
        with col2:
            st.metric(
                "Total Cost", 
                f"${st.session_state.total_cost:.4f}",
                help="Estimated OpenAI API cost"
            )
        
        st.metric(
            "Total Tokens", 
            f"{st.session_state.total_tokens:,}",
            help="Total tokens used (input + output)"
        )
        
        # Cost projection
        if st.session_state.total_queries > 0:
            avg_cost_per_query = st.session_state.total_cost / st.session_state.total_queries
            monthly_projection = avg_cost_per_query * 1000  # Assume 1000 queries/month
            
            st.markdown(f"""
            <div class="cost-tracker">
                <strong>💰 Cost Projection:</strong><br>
                Avg per query: ${avg_cost_per_query:.4f}<br>
                Monthly (1K queries): ${monthly_projection:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        # System Status
        st.subheader("🔧 System Status")
        check_system_status(api_url)
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Test API", use_container_width=True):
                test_api_connection(api_url)
        
        return api_url, model_choice, max_results, similarity_threshold

def display_example_questions():
    """Display example questions that users can click"""
    st.subheader("💡 Example Questions")
    
    example_questions = [
        ("💼", "What is the baggage weight limit for economy class?"),
        ("⚽", "Can I bring sports equipment on Turkish Airlines?"), 
        ("💰", "How much do excess baggage fees cost?"),
        ("🎒", "What are the carry-on baggage size restrictions?"),
        ("🐕", "Can I travel with my pet on Turkish Airlines?"),
        ("🎻", "What musical instruments can I bring on board?"),
        ("🧴", "What are the liquid restrictions in carry-on?"),
        ("✈️", "Do baggage rules differ for international flights?"),
        ("👶", "Are there special rules for baby strollers?"),
        ("🎿", "How do I transport ski equipment?")
    ]
    
    for emoji, question in example_questions:
        if st.button(f"{emoji} {question}", use_container_width=True, key=f"example_{hash(question)}"):
            handle_question(question, st.session_state.api_url, st.session_state.model_choice, 
                          st.session_state.max_results, st.session_state.similarity_threshold)

def handle_question(question, api_url, model, max_results, similarity_threshold):
    """Handle user question and get RAG response"""
    
    with st.spinner("🤖 Searching policies and generating answer..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Call API
            status_text.text("🔍 Searching relevant policies...")
            progress_bar.progress(25)
            
            response = requests.post(
                f"{api_url}/chat/openai",
                params={
                    "question": question,
                    "max_results": max_results,
                    "similarity_threshold": similarity_threshold,
                    "model": model
                },
                timeout=60  # Increased timeout for OpenAI
            )
            
            progress_bar.progress(50)
            status_text.text("🧠 Generating AI response...")
            
            if response.status_code == 200:
                progress_bar.progress(75)
                data = response.json()
                
                if data["success"]:
                    # Add to chat history
                    chat_entry = {
                        "timestamp": datetime.now(),
                        "question": question,
                        "answer": data["answer"],
                        "retrieved_docs": data.get("retrieved_docs", []),
                        "usage": data.get("usage_stats", {}),
                        "model": data.get("model_used", model),
                        "retrieval_stats": data.get("retrieval_stats", {}),
                        "context_used": data.get("context_used", False)
                    }
                    
                    st.session_state.chat_history.append(chat_entry)
                    
                    # Update statistics
                    st.session_state.total_queries += 1
                    
                    usage = data.get("usage_stats", {})
                    if "estimated_cost" in usage:
                        st.session_state.total_cost += usage["estimated_cost"]
                    if "total_tokens" in usage:
                        st.session_state.total_tokens += usage["total_tokens"]
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Answer generated successfully!")
                    
                    time.sleep(1)  # Brief pause to show completion
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Answer generated successfully!")
                    st.rerun()
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error: {data.get('error', 'Unknown error')}")
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            progress_bar.empty()
            status_text.empty()
            st.error("⏰ Request timed out. The AI is taking too long to respond. Please try again.")
        except requests.exceptions.RequestException as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Connection Error: {str(e)}")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Unexpected Error: {str(e)}")

def display_chat_history():
    """Display chat history with enhanced formatting"""
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="warning-box">
            <h4>💭 No questions asked yet!</h4>
            <p>Try asking something about Turkish Airlines baggage policies using the examples on the right, or type your own question above.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.subheader("📜 Chat History")
    
    # Display in reverse order (newest first)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(
            f"Q: {chat['question'][:60]}..." if len(chat['question']) > 60 else f"Q: {chat['question']}", 
            expanded=(i==0)  # Expand only the most recent
        ):
            
            # Question and metadata
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**🙋 Question:** {chat['question']}")
            with col2:
                st.markdown(f"**⏰ Asked:** {chat['timestamp'].strftime('%H:%M:%S')}")
            
            # Answer
            st.markdown("**🤖 AI Answer:**")
            st.markdown(f'<div class="chat-message">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            # Usage statistics
            st.markdown("**📊 Usage Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Used", chat.get("model", "Unknown"))
            with col2:
                tokens = chat.get("usage", {}).get("total_tokens", 0)
                st.metric("Tokens", f"{tokens:,}")
            with col3:
                cost = chat.get("usage", {}).get("estimated_cost", 0)
                st.metric("Cost", f"${cost:.4f}")
            with col4:
                context_used = "✅ Yes" if chat.get("context_used", False) else "❌ No"
                st.metric("Context Used", context_used)
            
            # Retrieved sources
            retrieved_docs = chat.get("retrieved_docs", [])
            if retrieved_docs:
                st.markdown("**📚 Sources Used:**")
                for j, doc in enumerate(retrieved_docs):
                    similarity = doc.get("similarity_score", 0)
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>📄 Source {j+1}:</strong> {doc.get("source", "Unknown")}<br>
                        <strong>🎯 Relevance:</strong> {similarity:.1%}<br>
                        <strong>📝 Content:</strong> {doc.get("content_preview", "N/A")}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ No relevant sources found</strong><br>
                    The AI answered based on general knowledge, not specific Turkish Airlines policies.
                </div>
                """, unsafe_allow_html=True)
            
            # Retrieval statistics
            retrieval_stats = chat.get("retrieval_stats", {})
            if retrieval_stats and retrieval_stats.get("total_retrieved", 0) > 0:
                st.markdown("**🔍 Search Quality:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Documents Found", retrieval_stats.get("total_retrieved", 0))
                with col2:
                    st.metric("Avg Similarity", f"{retrieval_stats.get('avg_similarity', 0):.1%}")
                with col3:
                    st.metric("Best Match", f"{retrieval_stats.get('max_similarity', 0):.1%}")
                with col4:
                    quality = retrieval_stats.get("context_quality", "unknown")
                    st.metric("Context Quality", quality.title())

def check_system_status(api_url):
    """Check and display system status"""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Overall status
            status = data.get("status", "unknown")
            if status == "healthy":
                st.success("🟢 System Online")
                st.session_state.api_status = "healthy"
            else:
                st.error("🔴 System Issues")
                st.session_state.api_status = "unhealthy"
            
            # Individual services
            rag_features = data.get("rag_features", {})
            
            services = {
                "Vector Search": rag_features.get("vector_operations", "unknown"),
                "Embeddings": rag_features.get("embedding_service", "unknown"),
                "OpenAI API": rag_features.get("openai_service", "unknown")
            }
            
            for service, service_status in services.items():
                if service_status == "available":
                    st.success(f"✅ {service}")
                else:
                    st.error(f"❌ {service}: {service_status}")
                    
            # Additional info
            db_info = data.get("data", {})
            if "total_policies" in db_info:
                st.info(f"📊 Database: {db_info['total_policies']} policies loaded")
                
            # OpenAI model info
            openai_model = rag_features.get("openai_model", "unknown")
            if openai_model != "unknown":
                st.info(f"🤖 OpenAI Model: {openai_model}")
                
        else:
            st.error("🔴 API Unreachable")
            st.session_state.api_status = "unreachable"
            
    except Exception as e:
        st.error(f"🔴 Connection Failed: {str(e)}")
        st.session_state.api_status = "error"

def test_api_connection(api_url):
    """Test API connection and display detailed results"""
    try:
        with st.spinner("Testing API connection..."):
            response = requests.get(f"{api_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                st.success("✅ API connection successful!")
                
                # Show detailed status
                with st.expander("📋 Detailed API Status"):
                    st.json(data)
                    
            else:
                st.error(f"❌ API returned status {response.status_code}")
                st.code(response.text)
    except Exception as e:
        st.error(f"❌ Connection test failed: {str(e)}")

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Question")
        
        # Question input
        user_question = st.text_area(
            "What would you like to know about Turkish Airlines baggage policies?",
            placeholder="Example: What is the baggage weight limit for economy class passengers on international flights?",
            height=120,
            key="question_input",
            help="Ask specific questions about baggage policies for the most accurate answers."
        )
        
        # Buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            ask_button = st.button("🚀 Ask AI Assistant", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("🎲 Random Example", use_container_width=True):
                example_questions = [
                    "What is the baggage weight limit for economy class?",
                    "Can I bring sports equipment on Turkish Airlines?",
                    "How much do excess baggage fees cost?",
                    "What are the carry-on baggage size restrictions?",
                    "Can I travel with my pet on Turkish Airlines?",
                    "What musical instruments can I bring on board?",
                    "Are there restrictions on liquids in carry-on baggage?",
                    "Do baggage rules differ for international vs domestic flights?",
                    "Can I bring medical equipment in my carry-on?",
                    "What happens if my baggage is overweight?"
                ]
                import random
                selected_question = random.choice(example_questions)
                st.session_state.question_input = selected_question
                st.rerun()
        
        with col_btn3:
            if st.button("📋 View API Docs", use_container_width=True):
                st.markdown("[Open API Documentation](http://localhost:8000/docs)")
        
        # Handle question submission
        if ask_button and user_question.strip():
            # Store current settings in session state for use in handle_question
            st.session_state.api_url = st.session_state.get('api_url', 'http://localhost:8000')
            st.session_state.model_choice = st.session_state.get('model_choice', 'gpt-3.5-turbo')
            st.session_state.max_results = st.session_state.get('max_results', 3)
            st.session_state.similarity_threshold = st.session_state.get('similarity_threshold', 0.3)
            
            handle_question(
                user_question, 
                st.session_state.api_url,
                st.session_state.model_choice, 
                st.session_state.max_results, 
                st.session_state.similarity_threshold
            )
        elif ask_button:
            st.warning("⚠️ Please enter a question before clicking Ask!")
        
        # Display chat history
        display_chat_history()
    
    with col2:
        display_example_questions()
    
    # Sidebar (this will update session state variables)
    api_url, model_choice, max_results, similarity_threshold = display_sidebar()
    
    # Update session state with sidebar values
    st.session_state.api_url = api_url
    st.session_state.model_choice = model_choice
    st.session_state.max_results = max_results
    st.session_state.similarity_threshold = similarity_threshold

if __name__ == "__main__":
    main()