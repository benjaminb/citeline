"""Quick test to check if NN model is producing valid outputs"""
import torch
import numpy as np

# Load the traced model
model = torch.jit.load("notebooks/best_model_traced.pth", map_location="cpu")
model.eval()

# Create a random unit vector (like our embeddings)
test_vector = np.random.randn(1024)
test_vector = test_vector / np.linalg.norm(test_vector)  # Normalize

print(f"Input vector:")
print(f"  Shape: {test_vector.shape}")
print(f"  Norm: {np.linalg.norm(test_vector):.6f}")
print(f"  First 5 values: {test_vector[:5]}")

# Transform with NN
with torch.no_grad():
    input_tensor = torch.tensor([test_vector], dtype=torch.float32)
    output = model(input_tensor).numpy()[0]

print(f"\nOutput vector:")
print(f"  Shape: {output.shape}")
print(f"  Norm: {np.linalg.norm(output):.6f}")
print(f"  First 5 values: {output[:5]}")
print(f"  Contains NaN: {np.isnan(output).any()}")
print(f"  Contains Inf: {np.isinf(output).any()}")

# Test with multiple vectors
print("\n=== Testing batch of 10 vectors ===")
batch = np.random.randn(10, 1024)
batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)

with torch.no_grad():
    input_tensor = torch.tensor(batch, dtype=torch.float32)
    output = model(input_tensor).numpy()

print(f"Output norms: {np.linalg.norm(output, axis=1)}")
print(f"All close to 1.0? {np.allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-5)}")
