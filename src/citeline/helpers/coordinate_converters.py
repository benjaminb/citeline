import numpy as np

"""
Reference: https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
"""


def euclidean_to_spherical(vector: np.ndarray) -> np.ndarray:
    """
    Converts a unit vector in euclidean coordinates to spherical coordinates.
    """
    dim = vector.shape[0]
    spherical = np.zeros(dim)
    r = np.linalg.norm(vector)
    if r == 0:
        raise ValueError(
            "Zero vector cannot be converted to spherical coordinates. Also, no embeddings or difference vectors should be zero."
        )
    spherical[0] = r
    for i in range(1, dim - 1):
        spherical[i] = np.atan2(np.linalg.norm(vector[i:]), vector[i - 1])
    spherical[-1] = np.arctan2(vector[-1], vector[-2])

    return spherical


def spherical_to_euclidean(vector: np.ndarray) -> np.ndarray:
    """
    Converts a vector in spherical coordinates to euclidean coordinates.
    Assumes the radius r = 1 (unit sphere).
    """
    dim = vector.shape[0]
    if dim < 2:
        raise ValueError("Spherical coordinates must have at least 2 dimensions.")

    r = vector[0]

    # Set up output vector as a product accumulator
    euclidean = np.full(dim, r)
    for i in range(dim - 1):
        for j in range(i):
            euclidean[i] *= np.sin(vector[j])
        euclidean[i] *= np.cos(vector[i + 1])
    euclidean[-1] *= np.prod(np.sin(vector[1:]))
    return euclidean


def main():
    # Tests
    for vec in [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0]), np.array([1.0, 1.0]) / np.sqrt(2)]:
        print("\n---\n")
        print("Original Euclidean coordinates:", vec)
        spherical = euclidean_to_spherical(vec)
        print("Spherical coordinates:", spherical)
        reconstructed = spherical_to_euclidean(spherical)
        print("Reconstructed Euclidean coordinates:", reconstructed)
        print("Reconstruction error:", np.linalg.norm(vec - reconstructed))


if __name__ == "__main__":
    main()
