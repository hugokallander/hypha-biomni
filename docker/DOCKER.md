# Biomni Docker Usage (docker/ layout)

Build command (note -f path):
```bash
docker build -t biomni-tools -f docker/Dockerfile .
```

Run (lists tools):
```bash
docker run --rm biomni-tools
```

Interactive shell:
```bash
docker run -it --entrypoint bash biomni-tools
```

List tools manually inside container:
```bash
micromamba run -n biomni_e1 python docker/list_tools.py
```

Extend for full E1 (R/CLI heavy stack): create a new Dockerfile FROM this image and replicate logic from `biomni_env/setup.sh`.
