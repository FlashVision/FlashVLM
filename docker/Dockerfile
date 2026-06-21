FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY pyproject.toml README.md ./
COPY flashvlm/ ./flashvlm/

RUN pip install --no-cache-dir -e ".[training]"

COPY configs/ ./configs/
COPY examples/ ./examples/

RUN mkdir -p /app/models /app/data /app/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import flashvlm; print(flashvlm.__version__)" || exit 1

ENTRYPOINT ["flashvlm"]
CMD ["--help"]
