# Vision Chat - Chainlit web UI image.
#
# Build:  docker build -t vision-chat .
# Run:    docker run -p 8000:8000 vision-chat
#
# The model server is NOT part of this image:
# point LLAMA_BASE_URL at a running llama.cpp server.

FROM python:3.11-slim-bookworm

# uv, pinned to the version that generated uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencies only (layer is cached until the lockfile changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# The project itself + UI assets
COPY . ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

RUN useradd --create-home app && chown --recursive app:app /app
USER app

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
