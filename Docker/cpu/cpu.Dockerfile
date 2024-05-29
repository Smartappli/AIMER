FROM python:3-slim-bullseye

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0
ENV PORT 8008

# Install necessary packages
RUN apt update && apt install -y --no-install-recommends libopenblas-dev ninja-build build-essential pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

# Create a non-root user
RUN useradd -m myuser

# Change to the non-root user
USER myuser

# Copy the application code
COPY --chown=myuser:myuser . .

# Install Python dependencies
RUN python -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with OpenBLAS)
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install 'llama-cpp-python[server]==0.2.76' --verbose

# Expose the port
EXPOSE 8008

# Run the server
CMD python3 -m llama_cpp.server --config_file config-cpu.json
