# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="GenEuclidean" \
      org.opencontainers.image.description="SE(3)-Equivariant 3D Molecular Encoding, Chemical NLP, NanoFold3 Co-Folding, Active Learning & Circuit Studio" \
      org.opencontainers.image.version="5.3.0" \
      org.opencontainers.image.source="https://github.com/daniil-777/geneuclidean" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md CITATION.cff CHANGELOG.md ./
COPY src ./src

# Install CPU PyTorch + core GenEuclidean SDK cleanly
RUN uv pip install --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch \
    && uv pip install --no-cache -e "."

ENTRYPOINT ["geneuclidean"]
CMD ["--help"]
