## Biomni tools Dockerfile (conda-based)
FROM continuumio/miniconda3:24.5.0-0

SHELL ["bash", "-lc"]
WORKDIR /app

# Copy only env file first for better layer caching
COPY biomni_env/environment.yml /tmp/environment.yml

# Create environment (name inside file is 'biomni')
RUN conda env update -n base -f /tmp/environment.yml \
    && conda clean -afy

# Ensure muscle (already in env) present; fallback install if missing
RUN if ! command -v muscle >/dev/null 2>&1; then conda install -y -c bioconda muscle || true; fi

# Copy project
COPY . /app

# Install package
RUN pip install --no-cache-dir .

ENV BIOMNI_DATA_PATH=/data
RUN mkdir -p /data
VOLUME ["/data"]

ENTRYPOINT ["python", "docker/list_tools.py"]
