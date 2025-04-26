#!/bin/sh

ollama serve &

pid=$!

echo "Waiting for Ollama to start..."
sleep 5

echo "Pulling model llama3.2:1b..."
ollama pull llama3.2:1b
echo "Model pulled."