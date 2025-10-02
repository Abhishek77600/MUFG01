FROM python:3.10-slim


WORKDIR /app


# Install OS deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
&& rm -rf /var/lib/apt/lists/*


# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy source
COPY src/ ./src
COPY artifacts/ ./artifacts


# Create data dir
RUN mkdir -p /app/data


EXPOSE 8000


# Default command: start API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]