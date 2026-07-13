FROM python:3.12-slim

WORKDIR /app

# COPY . . lands the repo root in /app, so the package itself sits at /app/app.
# Streamlit puts the SCRIPT's directory (/app/app) on sys.path, not the working dir,
# so `import app.agent.orchestrator` can't resolve. Celery doesn't hit this because it
# runs from the working dir. Pinning PYTHONPATH makes both entry points agree.
ENV PYTHONPATH=/app

# No build toolchain needed: psycopg2-binary ships prebuilt wheels, and every other
# dependency is pure Python or wheel-distributed. Dropping gcc keeps the image small.

# Copy requirements first, on its own layer. Docker caches layers, so as long as
# requirements.txt is unchanged, `pip install` is skipped on rebuild — even when
# application code changes. Copying everything up front would bust that cache
# on every single edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. See .dockerignore — .env and the local venvs are excluded,
# so no secrets or host-built binaries get baked into the image layers.
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
