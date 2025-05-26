FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY requirements.lock /app/requirements.lock

RUN uv pip install --system --no-cache -r requirements.lock

COPY src/ /app/src/

RUN uv pip install --system --no-cache -e .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=poll"]