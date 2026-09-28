# Development environment for this course: Python, pandas, scikit-learn, matplotlib,
# Jupyter and pytest all live inside this image. Nothing is installed on your machine.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCIKIT_LEARN_DATA=/data/scikit_learn_data

# A regular (non-root) user, so files you create from Jupyter aren't owned by root.
RUN useradd --create-home --uid 1000 learner \
    && mkdir -p /data/scikit_learn_data \
    && chown -R learner /data

WORKDIR /app

# Install the project and its tools. Only the files pip needs are copied here;
# the whole project folder is mounted over /app at run time (see compose.yaml).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev]" && chown -R learner /app

USER learner
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/app"]
