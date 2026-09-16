# Reproducibility container for the dementia-risk EHR pipeline.
#
# Scope (see DOCKER.md for the full explanation):
#   - Random Forest / XGBoost baselines (baselines/): fully verified inside
#     this image to run a real train -> test -> predict cycle on a mounted
#     data split.
#   - GNN, BERT-family models, explainability, risk stratification: included
#     and runnable from the same image/entrypoint, but NOT verified against
#     an actual `docker build && docker run` in the environment this image
#     was authored in (no Docker daemon / GPU was available there -- see
#     DOCKER.md). Treat those paths as "should work, please confirm" rather
#     than "confirmed".
#
# No trained model weights are baked into this image. Every checkpoint
# directory (baselines/checkpoints, gnn/checkpoints_gnn, bert_models/*
# checkpoints) is created fresh at container *runtime*, written to a
# directory you mount in from the host (see DOCKER.md / entrypoint.sh
# --help). Training a fresh model on every run -- rather than shipping
# pre-trained weights -- was a specific requirement for this container.

FROM python:3.10-slim

LABEL description="Reproducibility container: dementia risk prediction from longitudinal EHR data (UKDRI reproducibility submission)"

# System packages needed to build a couple of the pinned wheels
# (scikit-learn/xgboost/torch all ship manylinux wheels for this base image,
# but keep a minimal toolchain around in case pip has to build anything from
# source on an unusual host architecture).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached across code
# changes. See requirements.txt for the (empirically verified) torch /
# transformers version pinning rationale.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Now copy the actual codebase. .dockerignore excludes existing checkpoints,
# generated outputs, caches, and .git -- see that file for the exact list.
COPY . /app

# Default mount points for data in / results out. Nothing is written here at
# build time; these are created so `docker run -v host:/data -v host:/output`
# has somewhere to land even before the user's first run.
RUN mkdir -p /data /output

ENV DATA_DIR=/data \
    OUTPUT_DIR=/output \
    PYTHONUNBUFFERED=1

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["help"]
