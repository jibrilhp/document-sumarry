# Install uv
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc tesseract-ocr python3-filetype python3-magic libfile-libmagic-perl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app

COPY . .

# Install dependencies
RUN uv sync

# Run the application
CMD ["./.venv/bin/fastapi", "run", "main.py"]
