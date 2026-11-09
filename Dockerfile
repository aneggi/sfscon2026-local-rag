FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY local_rag ./local_rag
COPY app.py .
COPY config ./config

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]

FROM runtime AS tests
COPY requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests ./tests
CMD ["sh", "-c", "python -m pytest -q -p no:cacheprovider && ruff check --no-cache . && ruff format --check --no-cache ."]

# Keep the ordinary image small: test dependencies belong only to the tests target.
FROM runtime AS app
