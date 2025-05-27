#!/usr/bin/env python3
"""
Windows-compatible simple startup for debugging
"""
import subprocess
import time
import sys
import os

def main():
    print("🚀 SIMPLE STARTUP - DEBUG MODE")
    print("=" * 50)
    
    # Environment info
    print("🔍 Environment Check:")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  Python version: {sys.version}")
    print(f"  OpenAI API Key: {'✅ Present' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    print(f"  DB Host: {os.getenv('DB_HOST', 'localhost')}")
    
    # List available files
    print(f"  Python files: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    
    # Database wait (simplified)
    print("\n📊 Database Check:")
    db_ready = False
    for i in range(10):
        try:
            result = subprocess.run([
                'pg_isready', 
                '-h', os.getenv('DB_HOST', 'db'),
                '-U', os.getenv('DB_USER', 'postgres')
            ], capture_output=True, timeout=5)
            
            if result.returncode == 0:
                print("✅ Database is ready!")
                db_ready = True
                break
        except Exception as e:
            print(f"⚠️ Database check error: {e}")
        
        print(f"⏳ Waiting for database... ({i+1}/10)")
        time.sleep(2)
    
    if not db_ready:
        print("⚠️ Database not ready, but continuing...")
    
    # Critical imports test
    print("\n📦 Import Tests:")
    
    critical_imports = ['fastapi', 'uvicorn', 'asyncpg', 'myapp']
    all_imports_ok = True
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {str(e)}")
            all_imports_ok = False
    
    if not all_imports_ok:
        print("❌ Critical imports failed - cannot start server")
        sys.exit(1)
    
    # Optional: Quick web scraping (with short timeout)
    print("\n🌐 Web Scraping (optional):")
    try:
        result = subprocess.run(['python', 'web_scrapper.py'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Web scraping completed successfully")
        else:
            print("⚠️ Web scraping had issues:")
            print(f"    Return code: {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}...")
            print("    Continuing without scraping...")
    except subprocess.TimeoutExpired:
        print("⏰ Web scraping timeout - continuing...")
    except Exception as e:
        print(f"⚠️ Web scraping error: {e} - continuing...")
    
    # Start FastAPI server
    print("\n🎯 Starting FastAPI Server:")
    print("    Host: 0.0.0.0")
    print("    Port: 8000")
    print("    Access: http://localhost:8000")
    
    try:
        # Use simple uvicorn startup
        subprocess.run([
            'uvicorn', 'myapp:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--log-level', 'info'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()