ARG CUDA_IMAGE="12.4.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST=0.0.0.0
ENV PORT=8008

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
COPY --chown=myuser:myuser . .

# Setting build-related environment variables
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

# Install Python dependencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with CUDA)
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Expose the port
EXPOSE 8008

# Run the server
CMD ["python", "-m", "llama_cpp.server", "--config_file", "config-cpu.json"]
