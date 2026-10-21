# Research dashboard + API. Reads precomputed artifacts; never trains or downloads data.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch wheels keep the image smaller.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app artifacts/metrics ./artifacts/metrics
COPY --chown=app:app artifacts/figures ./artifacts/figures
COPY --chown=app:app artifacts/reports ./artifacts/reports
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "python -m dp_ecg.cli serve --host 0.0.0.0 --port ${PORT}"]
