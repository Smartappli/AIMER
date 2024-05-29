ARG CUDA_IMAGE="12.4.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0
ENV PORT 8008

# Install necessary packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    python3 python3-pip gcc wget git \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Create a non-root user
RUN useradd -m myuser

# Change to the non-root user
USER myuser

# Add .local/bin to PATH
ENV PATH="/home/myuser/.local/bin:${PATH}"

# Copy the application code
COPY --chown=myuser:myuser ../.. .

# Set build-related environment variables
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install dependencies
RUN python3 -m pip install --upgrade pip \
    && pip install pytest==8.2.1 cmake==3.29.3 \
    scikit-build==0.17.6 setuptools==70.0.0 \
    fastapi==0.111.0 uvicorn==0.30.0 \
    sse-starlette==2.1.0 pydantic-settings==2.2.1 \
    starlette-context==0.3.6

# Install llama-cpp-python (build with CUDA)
RUN CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install 'llama-cpp-python[server]==0.2.76' --verbose

# Expose the port
EXPOSE 8008

# Run the server
CMD python3 -m llama_cpp.server --config_file config-cuda.json
