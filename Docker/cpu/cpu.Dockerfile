FROM python:3-slim-bullseye

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST=0.0.0.0
ENV PORT=8008

# Install necessary packages
RUN apt update && apt install -y --no-install-recommends git libopenblas-dev ninja-build build-essential pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

# Copy the application code
COPY --chown=root:root --chmod=755 . .

# Create a non-root user
RUN useradd -m myuser

# Change to the non-root user
USER myuser

# Add .local/bin to PATH
ENV PATH="/home/myuser/.local/bin:${PATH}"

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install uv
RUN python -m uv pip install pytest==8.3.2 cmake==3.30.1 \ 
    scikit-build==0.18.0 setuptools==71.1.0 \
    fastapi==0.111.1 uvicorn==0.30.3 \
    sse-starlette==2.1.2 pydantic-settings==2.3.4 \
    starlette-context==0.3.6

# Install llama-cpp-python (build with OpenBLAS)
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install 'llama-cpp-python==0.2.83' --verbose

# Expose the port
EXPOSE 8008

# Run the server
CMD ["python", "-m", "llama_cpp.server", "--config_file", "config-cpu.json"]
