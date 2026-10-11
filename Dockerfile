FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --create-home appuser
COPY rag_app ./rag_app
COPY static ./static
COPY docs ./docs
COPY server.py ingest.py manage.py run_demo.py demo_queries.json ./
# COPY . ./
RUN mkdir -p /data && chown appuser:appuser /data
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
