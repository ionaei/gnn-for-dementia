# Reproducibility container for the dementia-risk EHR pipeline.
#
# Scope (see DOCKER.md for the full explanation):
#   - Every module in this package (baselines/, gnn/, explainability/,
#     risk_stratification/, bert_models/) is included and runnable from this
#     same image/entrypoint, driven by the synthetic smoke-test data that
#     ships in the image (data_prep/five_updated_synthetic.csv) or your own
#     mounted UK Biobank extract.
#   - What "verified" means for this specific image: the pip install layer
#     was verified in a clean virtualenv, and every entrypoint.sh subcommand
#     was dry-run tested against the current codebase (via the APP_ROOT
#     override -- see entrypoint.sh). An actual `docker build && docker run`
#     was NOT executed in the environment this image was authored in (no
#     Docker daemon was available there -- see DOCKER.md for the full
#     disclosure). Please do a first `docker build` + smoke-test run on your
#     end before relying on this for anything high-stakes.
#
# No trained model weights are baked into this image. Every checkpoint
# directory (baselines/checkpoints, gnn/checkpoints_gnn, bert_models/*
# checkpoints) is created fresh at container *runtime*, written to a
# directory you mount in from the host (see DOCKER.md / entrypoint.sh
# --help). Training a fresh model on every run -- rather than shipping
# pre-trained weights -- was a specific requirement for this container.

FROM python:3.11-slim

LABEL description="Reproducibility container: dementia risk prediction from longitudinal EHR data (NeurIPS 2025 TS4H workshop paper, UKDRI reproducibility submission)"

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
# Note: data_prep/five_updated_synthetic.csv IS included (unlike an earlier
# version of this image) so the Docker workflow has out-of-the-box parity
# with the top-level README's non-Docker Quickstart, which references this
# exact file. It's a small (~290KB), fully synthetic, already-git-committed
# dataset -- not a trained weight, so it doesn't fall under the
# no-baked-in-weights requirement above.
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
