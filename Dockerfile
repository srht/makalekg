FROM python:3.12-slim

ARG WITH_EMBED=0
# GLiNER (hızlı varlık modeli): torch CPU + gliner, imaja ~1 GB ekler. Kapatmak için WITH_GLINER=0.
ARG WITH_GLINER=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/hf \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KG_DATA_DIR=/app/data \
    KG_REPORTS_DIR=/app/reports

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_EMBED" = "1" ]; then pip install "pykeen>=1.10"; fi
RUN if [ "$WITH_GLINER" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install -r requirements-ml.txt; \
    fi

COPY . .
RUN useradd --create-home --uid 10001 kg \
 && mkdir -p /app/data /app/reports \
 && chown -R kg:kg /app
USER kg

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

# Tek işçi: iş kuyruğu süreç içinde tutulur
CMD ["uvicorn", "kg.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
