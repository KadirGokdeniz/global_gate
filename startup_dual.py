#!/usr/bin/env python3
import subprocess
import threading
import time
import sys

def start_fastapi():
    print("🚀 Starting FastAPI (FIXED)...")
    subprocess.run(['uvicorn', 'myapp:app', '--host', '0.0.0.0', '--port', '8000'])

def start_streamlit():
    print("⏳ Waiting 15 seconds...")
    time.sleep(15)
    print("🎨 Starting Streamlit...")
    subprocess.run(['streamlit', 'run', 'streamlit_app.py', '--server.port', '8501', '--server.address', '0.0.0.0', '--server.headless', 'true'])

def main():
    print("🔧 FIXED STARTUP")
    fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
    fastapi_thread.start()
    start_streamlit()

if __name__ == "__main__":
    main()
