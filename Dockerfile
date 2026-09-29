# syntax=docker/dockerfile:1
# Development environment for this course: Python, pandas, scikit-learn, matplotlib,
# JupyterLab and pytest all live inside this image. Nothing is installed on your machine.
FROM python:3.13-slim

# uv installs Python packages like pip, but downloads them in parallel and is much faster.
COPY --from=ghcr.io/astral-sh/uv:0.12.20 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    SCIKIT_LEARN_DATA=/data/scikit_learn_data

# A regular (non-root) user, so files you create from Jupyter aren't owned by root.
RUN useradd --create-home --uid 1000 learner \
    && mkdir -p /data/scikit_learn_data \
    && chown -R learner /data

WORKDIR /app

# Install the packages. This step depends ONLY on pyproject.toml, so editing docs,
# notebooks or code never triggers a re-download; only changing dependencies does.
# The real project folder is mounted over /app at run time (see compose.yaml), so
# empty placeholders for README.md and the package are enough here.
# The cache mount keeps downloaded packages between builds, so even a rebuild
# doesn't download everything again.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p src/house_price \
    && touch README.md src/house_price/__init__.py \
    && uv pip install --system -e ".[dev]" \
    && chown -R learner /app

USER learner
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/app"]
