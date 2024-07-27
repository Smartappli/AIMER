ARG CUDA_IMAGE="12.5.0-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST=0.0.0.0
ENV PORT=8008

# Install necessary packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Copy the application code
COPY --chown=root:root --chmod=755 ../.. .

# Create a non-root user
RUN useradd -m myuser

# Change to the non-root user
USER myuser

# Add .local/bin to PATH
ENV PATH="/home/myuser/.local/bin:${PATH}"

# Set build-related environment variables
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install dependencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with CUDA)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Expose the port
EXPOSE 8008

# Run the server
CMD ["python3", "-m", "llama_cpp.server", "--config_file", "config-cuda.json"]
