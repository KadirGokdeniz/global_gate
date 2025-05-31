# ===============================================
# stop.sh - Clean shutdown script
# ===============================================
#!/bin/bash

echo "🛑 Stopping Turkish Airlines System"
echo "==================================="

echo "🔄 Stopping services gracefully..."
docker-compose down

echo "🗑️ Cleaning up containers..."
docker-compose down --remove-orphans

echo "💾 Database data preserved in volume"
echo "✅ System stopped cleanly"
