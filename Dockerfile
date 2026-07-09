# syntax=docker/dockerfile:1
#
# Deployable image for the HR Assistant Streamlit UI.
#
# Runs in FALLBACK mode by default (deterministic canned HR responses), which
# needs no GPU and works on any Linux host such as Azure Container Apps.
# See docs/AZURE_DEPLOYMENT.md for how to build, push, and deploy this image,
# and for the (GPU) path required to serve the fine-tuned model instead.

FROM python:3.11-slim

# --- Runtime env -----------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HR_FALLBACK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# --- Python dependencies (slim, fallback-mode) -----------------------------
# Copy only the requirements first to maximize Docker layer caching.
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# --- Application code -------------------------------------------------------
COPY .streamlit/ ./.streamlit/
COPY src/ ./src/

# Run as a non-root user (good practice; required by some Azure policies).
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit exposes a health endpoint at /_stcore/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else sys.exit(1)" || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
