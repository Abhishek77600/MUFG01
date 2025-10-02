# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for models if needed
RUN mkdir -p /app/models

# Expose port for Streamlit
EXPOSE 8504

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8504/healthz || exit 1

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8504", "--server.address=0.0.0.0", "--server.headless=true"]