# Stage 1: Build stage to download the model
FROM ollama/ollama:latest AS builder

# NVIDIA Container Toolkit setup (optional?)
# RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
#     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# RUN curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
#     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
#     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# RUN sudo apt-get update

# RUN sudo apt-get install -y nvidia-container-toolkit

# Run ollama serve in the background, pull the model, then stop the server
# This happens during the image build process
RUN \
    ollama serve & \
    pid=$! && \
    echo "Waiting for Ollama server to start..." && \
    sleep 10 && \
    echo "Pulling llama3.2:1b model..." && \
    ollama pull llama3.2:1b && \
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

# The default entrypoint for ollama/ollama image is to run 'ollama serve'.
# If you override ENTRYPOINT/CMD, make sure it runs 'ollama serve'.
# For example, if you still need your entrypoint script for other reasons,
# modify entrypoint.sh to ONLY run 'ollama serve'.
# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh
# ENTRYPOINT [ "/entrypoint.sh" ] 
# If you don't need a custom script, you can rely on the base image's default
# or explicitly set it:
ENTRYPOINT [ "ollama", "serve" ]
# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh

# ENTRYPOINT [ "/entrypoint.sh" ]

# EXPOSE 11434