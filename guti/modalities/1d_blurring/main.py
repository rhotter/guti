import numpy as np
from guti.data_utils import save_svd, Parameters
from guti.svd import compute_svd_cpu

# Physical parameters
L = 0.1  # Length of domain in meters (10 cm)
sigma = 0.01  # Width of convolution kernel in meters (1 cm)


def create_convolution_matrix(input_dim, output_dim):
    """Create the convolution matrix for given input and output discretization."""
    dx = L / input_dim
    x_in = np.linspace(0, L, input_dim)
    x_out = np.linspace(0, L, output_dim)

    # Create convolution kernel matrix
    K = np.zeros((output_dim, input_dim))
    for i in range(output_dim):
        for j in range(input_dim):
            # Gaussian kernel
            K[i, j] = np.exp(-((x_out[i] - x_in[j]) ** 2) / (2 * sigma**2))

    # Normalize the kernel
    K *= dx / (sigma * np.sqrt(2 * np.pi))
    return K


num_voxels = 128
A = create_convolution_matrix(num_voxels, num_voxels)
s = compute_svd_cpu(A)
save_svd(s, "1d_blurring", Parameters(num_voxels=num_voxels))
