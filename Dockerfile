FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir openenv-core httpx pydantic uvicorn fastapi openai

COPY . .

ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=4
ENV MAX_CONCURRENT_ENVS=100
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

CMD uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS
