import numpy as np

"""
Reference: https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
"""


def euclidean_to_spherical(vector: np.ndarray) -> np.ndarray:
    """
    Converts a unit vector in euclidean coordinates to spherical coordinates.
    Using ata2, all angles should be in the range [-pi, pi].
    The first coordinate is the radius r, which should be 1 for unit vectors.
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

def add_spherical(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Takes vectors a and b in spherical coordinates and returns their vector sum in spherical coordinates.
    This is done by adding all the angular components in a and b (assumed to be all but the first). 
    This sets the radius to 1, to avoid floating imprecision.

    It also mods the angles to be in the range [-pi, pi], which is the range used by atan2.
    
    """
    # Check that all angular components are in [-pi, pi] and vectors have same dims
    if np.any(a[1:] < -np.pi) or np.any(a[1:] > np.pi):
        raise ValueError("All angular components of a must be in the range [-pi, pi]. Vector a has component(s) out of range.")
    if np.any(b[1:] < -np.pi) or np.any(b[1:] > np.pi):
        raise ValueError("All angular components of b must be in the range [-pi, pi]. Vector b has component(s) out of range.")
    if a.shape != b.shape:
        raise ValueError("Vectors a and b must have the same shape.")
    
    diff = np.zeros_like(a)
    diff[0] = 1.0  # radius
    diff[1:] = (a[1:] + b[1:] + np.pi) % (2 * np.pi) - np.pi
    return diff

def main():
    # Tests
    x = np.array([np.cos(np.pi / 10), np.sin(np.pi / 10)])
    y = np.array([np.cos(99 * np.pi / 100), np.sin(99 * np.pi / 100)])
    x_spherical, y_spherical = euclidean_to_spherical(x), euclidean_to_spherical(y)
    print(f"x in spherical coordinates: {x_spherical}")
    print(f"y in spherical coordinates: {y_spherical}")
    diff = y_spherical - x_spherical
    print(f"Difference in spherical coordinates: {diff}")
    y_reconstructed_spherical = x_spherical + diff
    print(f"Reconstructed y in spherical coordinates: {y_reconstructed_spherical}")
    y_reconstructed = spherical_to_euclidean(y_reconstructed_spherical)
    print(f"Reconstructed y in euclidean coordinates: {y_reconstructed}")
    print(f"Reconstruction error: {np.linalg.norm(y - y_reconstructed)}")
    
    # for vec in [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0]), np.array([1.0, 1.0]) / np.sqrt(2)]:
    #     print("\n---\n")
    #     print("Original Euclidean coordinates:", vec)
    #     spherical = euclidean_to_spherical(vec)
    #     print("Spherical coordinates:", spherical)
    #     reconstructed = spherical_to_euclidean(spherical)
    #     print("Reconstructed Euclidean coordinates:", reconstructed)
    #     print("Reconstruction error:", np.linalg.norm(vec - reconstructed))


if __name__ == "__main__":
    main()
