# ===============================================
# start.sh - Ana startup script
# ===============================================
#!/bin/bash

set -e

echo "🚀 Turkish Airlines System Startup"
echo "=================================="

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local wait_time=2
    
    echo "⏳ Checking $service health..."
    
    for i in $(seq 1 $max_attempts); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "✅ $service is healthy"
            return 0
        fi
        echo "🔄 Attempt $i/$max_attempts failed, waiting ${wait_time}s..."
        sleep $wait_time
    done
    
    echo "❌ $service health check failed"
    return 1
}

# Start all services
echo "📊 Starting database..."
docker-compose up -d db

echo "⏳ Waiting for database to be ready..."
check_service "Database" "http://localhost:5432"

echo "🕷️ Starting data scraper..."
docker-compose up scraper

echo "🔍 Checking scraper results..."
if [ $? -eq 0 ]; then
    echo "✅ Data scraping completed successfully"
else
    echo "❌ Data scraping failed"
    exit 1
fi

echo "🚀 Starting FastAPI backend..."
docker-compose up -d api

echo "⏳ Waiting for API to be ready..."
check_service "FastAPI" "http://localhost:8000/health"

echo "🎨 Starting Streamlit frontend..."
docker-compose up -d frontend

echo "⏳ Waiting for frontend to be ready..."
check_service "Streamlit" "http://localhost:8501/_stcore/health"

echo ""
echo "🎉 ALL SERVICES READY!"
echo "======================"
echo "🌐 FastAPI: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🎨 Streamlit: http://localhost:8501"
echo "🗄️ Database: localhost:5432"
echo ""
echo "🔧 Management commands:"
echo "  docker-compose logs [service]    # View logs"
echo "  docker-compose ps                # Service status"
echo "  docker-compose down              # Stop all"
echo "  docker-compose restart [service] # Restart service"