## Biomni tools Dockerfile (micromamba-based)
FROM mambaorg/micromamba:1.5.7

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ARG INSTALL_EXTRAS=1
ARG GPU=0
WORKDIR /app

# The official image defines MAMBA_USER (default 'mambauser'); capture it
ARG MAMBA_USER
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential g++ gcc wget git \
    && rm -rf /var/lib/apt/lists/*
# Create data directory early while root (avoid later root shell needing micromamba wrapper)
RUN mkdir -p /data && chown ${MAMBA_USER}:${MAMBA_USER} /data
USER ${MAMBA_USER}

## --- Layered environment setup ---
# Copy environment specs for caching
COPY --chown=${MAMBA_USER}:${MAMBA_USER} biomni_env/environment.base.yml /tmp/environment.base.yml
COPY --chown=${MAMBA_USER}:${MAMBA_USER} biomni_env/environment.extras.yml /tmp/environment.extras.yml
COPY --chown=${MAMBA_USER}:${MAMBA_USER} biomni_env/environment.yml /tmp/environment.legacy.yml

# Install base environment
RUN micromamba install -y -n base -f /tmp/environment.base.yml && \
    micromamba clean --all --yes

# Optionally install extras (heavy) if INSTALL_EXTRAS=1
RUN if [ "$INSTALL_EXTRAS" = "1" ]; then \
      micromamba install -y -n base -f /tmp/environment.extras.yml && \
      micromamba clean --all --yes ; \
    fi

# Optional GPU enablement (default CPU-only). Only adds CUDA stack when GPU=1.
RUN if [ "$GPU" = "1" ]; then \
      micromamba install -y -n base pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia && \
      micromamba clean --all --yes ; \
    fi

# Copy project source
COPY --chown=${MAMBA_USER}:${MAMBA_USER} . /app

# Install package into base env
RUN pip install --no-cache-dir .

# Optionally remove a subset of pip extras when INSTALL_EXTRAS=0 (noop if not installed)
RUN if [ "$INSTALL_EXTRAS" = "0" ]; then \
        pip uninstall -y DeepPurpose tdc FlowCytometryTools || true; \
    fi

ENV PATH=/opt/conda/envs/base/bin:/opt/conda/bin:$PATH \
    BIOMNI_DATA_PATH=/data \
    HOME=/home/${MAMBA_USER} \
    XDG_CACHE_HOME=/home/${MAMBA_USER}/.cache \
    MAMBA_ROOT_PREFIX=/opt/conda
RUN mkdir -p /home/${MAMBA_USER}/.cache/mamba/proc /home/${MAMBA_USER}/.cache/pip && \
    chown -R ${MAMBA_USER}:${MAMBA_USER} /home/${MAMBA_USER}/.cache && \
    chmod -R 0777 /home/${MAMBA_USER}/.cache
VOLUME ["/data"]

# Default command runs tool listing; can override with other scripts
ENTRYPOINT ["micromamba", "run", "-n", "base", "python"]
CMD ["-m", "biomni.hypha_service", "remote"]
