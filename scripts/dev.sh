# ===============================================
# dev.sh - Development startup
# ===============================================
#!/bin/bash

echo "🔧 Development Mode Startup"
echo "==========================="

# Only start database for development
docker-compose up -d db

echo "⏳ Waiting for database..."
sleep 10

echo "🏃 Starting development servers..."
echo "📝 Run these in separate terminals:"
echo ""
echo "  # Terminal 1 - Run scraper once:"
echo "  python scraper_only.py"
echo ""
echo "  # Terminal 2 - FastAPI with hot reload:"
echo "  uvicorn myapp:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "  # Terminal 3 - Streamlit with hot reload:"
echo "  streamlit run streamlit_app.py --server.port 8501"
echo ""
echo "🔧 Database ready at: localhost:5432"
