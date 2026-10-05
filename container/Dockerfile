FROM python:3.12-slim

# Triton compiles a small C launcher at runtime, so the image needs a C compiler.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# Versions from strands-decider docs/inference.md + training/README.md (CUDA setup).
RUN pip install \
      strands-decider==0.1.0 \
      transformers==5.17.0 \
      peft==0.21.0 \
      flash-linear-attention==0.5.2 \
      fla-core==0.5.2 \
 && python -c "import importlib.util, torch, triton; assert importlib.util.find_spec('fla'), 'fla missing'; print('torch', torch.__version__, 'triton', triton.__version__)"
# Note: don't `import fla` at build time. It probes for a GPU driver and fails without one.

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton
WORKDIR /app
COPY batching.py serve.py ./
EXPOSE 8080
# SageMaker runs the container with the argument "serve".
ENTRYPOINT ["python", "/app/serve.py"]
