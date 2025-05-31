# ===============================================
# reset.sh - Complete reset script
# ===============================================
#!/bin/bash

echo "🔄 COMPLETE SYSTEM RESET"
echo "========================"
echo "⚠️ This will delete ALL DATA!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Reset cancelled"
    exit 1
fi

echo "🛑 Stopping all services..."
docker-compose down

echo "🗑️ Removing volumes (ALL DATA LOST)..."
docker-compose down --volumes

echo "🔧 Rebuilding images..."
docker-compose build --no-cache

echo "🚀 Starting fresh system..."
./start.sh

echo "✅ System reset complete"
