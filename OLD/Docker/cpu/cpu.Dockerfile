FROM python:3-slim-trixie

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST=0.0.0.0
ENV PORT=8008

# Install necessary packages
RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Create a non-root user
RUN useradd -m myuser

# Change to the non-root user
USER myuser

# Add .local/bin to PATH
ENV PATH="/home/myuser/.local/bin:${PATH}"

# Copy the application code
COPY --chown=myuser:myuser . .

# Install Python dependencies
RUN python -m pip install --upgrade pip \
    && pip install pytest==8.2.1 cmake==3.23.3 \
    scikit-build==0.18.1 setuptools==80.9.0 \
    fastapi==0.121.0 uvicorn==0.38.0 \
    sse-starlette==3.0.3 pydantic-settings==2.11.0 \
    starlette-context==0.4.0

# Install llama-cpp-python (build with OpenBLAS)
RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama_cpp_python --verbose

# Expose the port
EXPOSE 8008

# Run the server
CMD ["python3", "-m", "llama_cpp.server", "--config_file", "config-cpu.json"]
