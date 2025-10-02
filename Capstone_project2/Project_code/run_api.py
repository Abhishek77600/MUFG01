#!/usr/bin/env python3
"""
Script to run the Heart Disease Classification API locally using uvicorn.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8002,
        reload=True  # Enable auto-reload for development
    )