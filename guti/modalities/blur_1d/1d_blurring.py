# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Physical parameters
L = 0.1  # Length of domain in meters (10 cm)
sigma = 0.01  # Width of convolution kernel in meters (1 cm)


def create_convolution_matrix(N):
    """Create the convolution matrix for given discretization N."""
    dx = L / N
    x = np.linspace(0, L, N)

    # Create convolution kernel matrix
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Gaussian kernel
            K[i, j] = np.exp(-((x[i] - x[j]) ** 2) / (2 * sigma**2))

    # Normalize the kernel
    K *= dx / (sigma * np.sqrt(2 * np.pi))
    return K


# %%
# Analyze for N=128
N1 = 128
K1 = create_convolution_matrix(N1)
U1, s1, Vh1 = linalg.svd(K1)

# %%
# Analyze for N=256
N2 = 256
K2 = create_convolution_matrix(N2)
U2, s2, Vh2 = linalg.svd(K2)

# %%
# Plot the singular values
plt.figure(figsize=(10, 6))
plt.semilogy(s1, "b.-", label=f"N={N1}")
plt.semilogy(s2, "r.-", label=f"N={N2}")

# Add vertical line at 1 cm mark (L/sigma points)
physical_index = int(L / sigma)  # Number of points corresponding to 1 cm
plt.axvline(x=physical_index, color="k", linestyle="--", label="1 cm scale")

plt.grid(True)
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.title("Singular Values of 1D Convolution Operator")
plt.legend()
plt.show()

# %%
# Plot the first few singular vectors for N=128
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    x = np.linspace(0, L, N1)
    plt.plot(x, U1[:, i], "b-", label=f"Left SV {i+1}")
    plt.plot(x, Vh1[i, :], "r--", label=f"Right SV {i+1}")
    plt.title(f"Singular Vector {i+1}, σ={s1[i]:.2e}")
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()
