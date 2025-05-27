#!/usr/bin/env python3
import subprocess
import threading
import time
import sys
import os

def start_fastapi():
    """Start FastAPI server"""
    print("🚀 Starting FastAPI server on port 8000...")
    try:
        # Run the existing startup sequence
        exec(open('startup.py').read())
    except Exception as e:
        print(f"❌ FastAPI startup error: {e}")
        # Fallback: direct uvicorn start
        subprocess.run(['uvicorn', 'myapp:app', '--host', '0.0.0.0', '--port', '8000'])

def start_streamlit():
    """Start Streamlit app"""
    print("⏳ Waiting for FastAPI to initialize...")
    time.sleep(10)  # Wait for FastAPI to be ready
    
    print("🎨 Starting Streamlit interface on port 8501...")
    subprocess.run([
        'streamlit', 'run', 'streamlit_app.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ])

def main():
    print("=" * 60)
    print("🇹🇷 TURKISH AIRLINES RAG SYSTEM - DUAL SERVICE STARTUP")
    print("=" * 60)
    
    # Start FastAPI in background thread
    fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Start Streamlit in main thread (this keeps the container running)
    try:
        start_streamlit()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()