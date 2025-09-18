# Stage 1: Build stage to download the model
FROM ollama/ollama:latest AS builder

ARG MODEL_NAME=deepseek-r1:70b

# Run ollama serve in the background, pull the model, then stop the server
# This happens during the image build process
RUN \
    ollama serve & \
    pid=$! && \
    echo "Waiting for Ollama server to start..." && \
    sleep 10 && \
    echo "pulling model ${MODEL_NAME}}" && \
    ollama pull ${MODEL_NAME} && \
    echo "Model pull complete. Stopping server..." && \
    kill $pid && \
    # Wait for the process to exit, ignore error if already stopped
    wait $pid || true

# Stage 2: Final image
FROM ollama/ollama:latest

# Copy the downloaded models from the builder stage
# Ollama stores models in /root/.ollama by default in the container
COPY --from=builder /root/.ollama /root/.ollama

# Expose the default Ollama port
EXPOSE 11434

ENTRYPOINT [ "ollama", "serve" ]
