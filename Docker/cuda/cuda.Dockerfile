ARG CUDA_IMAGE="13.1.1-devel-ubuntu24.04"
FROM nvidia/cuda:${CUDA_IMAGE}

ENV HOST=0.0.0.0

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 python3-venv python3-dev \
        git build-essential \
        cmake ninja-build \
        gcc g++ wget \
        ocl-icd-opencl-dev opencl-headers clinfo \
        libclblast-dev libopenblas-dev \
        libgomp1 \
    && mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if [ -f /usr/local/cuda/compat/libcuda.so.1 ]; then \
        ln -sf /usr/local/cuda/compat/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; \
    elif [ -f /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
        ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1; \
    fi

WORKDIR /app
COPY . .

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade --no-cache-dir pip wheel && \
    pip install --no-cache-dir \
        pytest scikit-build setuptools==80.9.0 \
        fastapi==0.128.0 uvicorn[standard]==0.4.0 sse-starlette \
        pydantic-settings starlette-context

# Installer llama-cpp-python avec CUDA
# On peut ajuster CMAKE_CUDA_ARCHITECTURES à ton GPU (80 = A100, 86 = RTX 30xx, 89 = RTX 40xx, etc.)
ENV GGML_CUDA=1
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90"
RUN pip install --no-cache-dir --verbose llama-cpp-python

CMD ["python3", "-m", "llama_cpp.server", "--config_file", "config-cuda.json"]
