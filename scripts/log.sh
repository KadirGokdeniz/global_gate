# ===============================================
# logs.sh - Easy log viewing
# ===============================================
#!/bin/bash

case $1 in
    "db"|"database")
        docker-compose logs -f db
        ;;
    "scraper"|"scrape")
        docker-compose logs scraper
        ;;
    "api"|"fastapi")
        docker-compose logs -f api
        ;;
    "frontend"|"streamlit")
        docker-compose logs -f frontend
        ;;
    "all"|"")
        docker-compose logs -f
        ;;
    *)
        echo "Usage: ./logs.sh [db|scraper|api|frontend|all]"
        echo ""
        echo "Available services:"
        echo "  db        - Database logs"
        echo "  scraper   - Data scraping logs"
        echo "  api       - FastAPI backend logs"
        echo "  frontend  - Streamlit frontend logs"
        echo "  all       - All services (default)"
        ;;
esac