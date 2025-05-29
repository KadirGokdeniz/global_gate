#!/usr/bin/env python3
"""
startup.py - Simple FastAPI startup script
Called by startup_dual.py
"""

import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Start FastAPI server"""
    logger.info("🚀 Starting FastAPI server...")
    
    try:
        uvicorn.run(
            "myapp:app",
            host="0.0.0.0", 
            port=8000,
            reload=False,  # Disable reload in production
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ FastAPI startup error: {e}")
        raise

if __name__ == "__main__":
    main()