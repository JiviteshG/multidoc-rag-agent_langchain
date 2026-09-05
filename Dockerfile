# 1. Python image
FROM python:3.13-slim

# 2. Set environment variables to optimize Python for containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system-level dependencies for FAISS and PDF processing
# Added libxml2-dev and libxslt-dev for lxml, and libmagic-dev for file processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    libmagic-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements.txt from local folder to the container
COPY requirements.txt .

# 6. Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of project code
COPY . .

# 8. Docker port Streamlit uses
EXPOSE 8501

# 9. Health check — Streamlit exposes a built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 10. The command to launch app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]