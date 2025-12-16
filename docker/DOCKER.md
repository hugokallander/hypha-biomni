# Biomni Docker Usage (docker/ layout)

Build command (note -f path):
```bash
docker build -t biomni-tools -f docker/Dockerfile .
```

Apple Silicon (M1/M2/M3) note:
- If you want to match a typical Kubernetes linux/amd64 deployment, build with:
	```bash
	docker buildx build --platform linux/amd64 -t biomni-tools -f docker/Dockerfile --load .
	```
	- Important: if your buildx builder uses the `docker-container` driver (common), you typically need `--load`
	  to make the image show up in `docker image ls` for `docker run`. Otherwise, Docker will say it “cannot find image”
	  and will try to pull it from a registry.
- If you build natively on macOS (linux/arm64), the Dockerfile will install PyTorch via pip CPU wheels
	because conda packages for `torchvision`/`torchaudio` may be unavailable on arm64.
	```bash
	docker buildx build --platform linux/arm64 -t biomni-tools:dev -f docker/Dockerfile --build-arg INSTALL_EXTRAS=1 --build-arg GPU=0 --load .
	```

GPU build:
```bash
docker build -t biomni-tools-gpu -f docker/Dockerfile . --build-arg GPU=1
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
