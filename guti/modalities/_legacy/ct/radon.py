"""
Radon transform matrix construction for CT imaging
================================================

This module provides utilities for building the system matrix for the Radon transform,
which represents how each pixel in the image contributes to each detector measurement
in a parallel-beam CT system.
"""

# %%
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csr_matrix


def build_radon_matrix(
    image_size: int,
    n_angles: int,
    detector_size: int | None = None,
) -> np.ndarray:
    """
    Build the system matrix for the Radon transform in parallel-beam CT.
    
    This matrix represents how each pixel in the image contributes to each detector
    measurement. The matrix shape is (n_angles * detector_size, image_size * image_size).
    
    Parameters
    ----------
    image_size : int
        Size of the square image (number of pixels in each dimension)
    n_angles : int
        Number of projection angles spanning 180 degrees
    detector_size : int, optional
        Number of detector elements. If None, uses image_size * sqrt(2) rounded up
        to ensure the entire image is covered at all angles.
        
    Returns
    -------
    np.ndarray
        System matrix of shape (n_angles * detector_size, image_size * image_size)
        where each row represents a single detector measurement and each column
        represents a pixel in the image.
    """
    if detector_size is None:
        # Ensure detector covers the entire image at all angles
        detector_size = int(np.ceil(image_size * np.sqrt(2)))
    
    # Create coordinate grids for the image
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    
    # Create coordinate grid for the detector
    detector_positions = np.linspace(-1, 1, detector_size)
    
    # Create angle grid
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    
    # Initialize the system matrix
    matrix_size = n_angles * detector_size
    image_pixels = image_size * image_size
    A = np.zeros((matrix_size, image_pixels))
    
    # Build the matrix row by row
    for i, angle in enumerate(angles):
        # Calculate projection coordinates for this angle
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # For each detector position
        for j, detector_pos in enumerate(detector_positions):
            row_idx = i * detector_size + j
            
            # Calculate the contribution of each pixel to this detector measurement
            # This is the line integral along the ray
            ray = detector_pos - (X * cos_theta + Y * sin_theta)
            
            # Use a Gaussian kernel to approximate the line integral
            # This creates a smooth transition between pixels
            sigma = 1.0 / image_size  # Adjust this parameter to control the width of the line
            weights = np.exp(-0.5 * (ray / sigma) ** 2)
            weights = weights / np.sum(weights)  # Normalize
            
            # Store the weights in the matrix
            A[row_idx, :] = weights.flatten()
    
    return A


def apply_radon_matrix(
    A: np.ndarray,
    image: np.ndarray,
    image_size: int,
    n_angles: int,
    detector_size: int | None = None,
) -> np.ndarray:
    """
    Apply the Radon transform matrix to an image.
    
    Parameters
    ----------
    A : np.ndarray
        The Radon transform matrix
    image : np.ndarray
        Input image of shape (image_size, image_size)
    image_size : int
        Size of the square image
    n_angles : int
        Number of projection angles
    detector_size : int, optional
        Number of detector elements
        
    Returns
    -------
    np.ndarray
        Sinogram of shape (detector_size, n_angles)
    """
    if detector_size is None:
        detector_size = int(np.ceil(image_size * np.sqrt(2)))
    
    # Flatten the image
    x = image.flatten()
    
    # Apply the matrix
    sino_flat = A @ x
    
    # Reshape to sinogram format
    return sino_flat.reshape(n_angles, detector_size).T


def compute_svd_spectrum(
    image_size: int,
    n_angles: int,
    detector_size: int | None = None,
    do_3d: bool = False,
) -> np.ndarray:
    """
    Compute the singular value spectrum of the Radon transform matrix.
    
    Parameters
    ----------
    image_size : int
        Size of the square image
    n_angles : int
        Number of projection angles
    detector_size : int, optional
        Number of detector elements
    do_3d : bool, optional
        Whether to compute the 3D Radon transform matrix
    Returns
    -------
    np.ndarray
        Array of singular values
    """
    # Build the matrix
    if do_3d:
        A = build_radon_matrix_3d((image_size, image_size, image_size), n_angles, (detector_size, detector_size))
    else:
        A = build_radon_matrix(image_size, n_angles, detector_size)
    print(A.shape)
    
    # Compute SVD
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    
    return s


def plot_svd_spectra(
    image_sizes: list[int] = [5, 15],
    n_angles: int = 180,
    detector_size: int | None = 10,
    do_3d: bool = False,
) -> None:
    """
    Compute and plot SVD spectra for different image sizes.
    
    Parameters
    ----------
    image_sizes : list of int
        List of different image sizes to compare
    n_angles : int
        Number of projection angles
    detector_size : int, optional
        Number of detector elements
    do_3d : bool, optional
        Whether to compute the 3D Radon transform matrix
    """
    plt.figure(figsize=(10, 6))
    
    for size in image_sizes:
        # Compute SVD spectrum
        s = compute_svd_spectrum(size, n_angles, detector_size, do_3d)
        
        # Plot singular values
        plt.semilogy(s, label=f'{size}x{size} image')
    
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Radon Transform Singular Value Spectra for Different Image Sizes')
    plt.grid(True)
    plt.legend()
    plt.show()


def demo_radon_matrix():
    """
    Demonstrate the Radon transform matrix construction and application.
    """
    # Parameters
    image_size = 32
    n_angles = 180
    detector_size = 45
    
    # Build the matrix
    A = build_radon_matrix(image_size, n_angles, detector_size)
    
    # Create a test image (a simple circle)
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    test_image = (X**2 + Y**2 < 0.5).astype(float)
    
    # Apply the transform
    sinogram = apply_radon_matrix(A, test_image, image_size, n_angles, detector_size)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.imshow(test_image, cmap='gray')
    ax1.set_title('Test Image')
    ax1.axis('off')
    
    ax2.imshow(sinogram, cmap='gray', aspect='auto')
    ax2.set_title('Sinogram')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def build_radon_matrix_3d(
    volume_shape: Tuple[int, int, int],
    n_angles: int,
    det_shape: Tuple[int, int] | None = None,
) -> csr_matrix:
    """
    Build the system matrix for the 3D Radon transform in parallel-beam CT.
    
    This matrix represents how each voxel in the volume contributes to each detector
    measurement. The matrix shape is (n_angles * nu * nv, nx * ny * nz).
    
    Parameters
    ----------
    volume_shape : Tuple[int, int, int]
        Shape of the volume (nx, ny, nz)
    n_angles : int
        Number of projection angles spanning 180 degrees
    det_shape : Tuple[int, int], optional
        Shape of the detector (nu, nv). If None, uses sqrt(2) * volume dimensions
        to ensure the entire volume is covered at all angles.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        System matrix of shape (n_angles * nu * nv, nx * ny * nz)
        where each row represents a single detector measurement and each column
        represents a voxel in the volume.
    """
    nx, ny, nz = volume_shape
    
    # Set detector dimensions if not provided
    if det_shape is None:
        nu = int(np.ceil(np.sqrt(2) * nx))
        nv = int(np.ceil(np.sqrt(2) * ny))
    else:
        nu, nv = det_shape
    
    # Create coordinate grids for the volume
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    
    # Flatten and stack coordinates
    vox_centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Create coordinate grids for the detector
    detector_u = np.linspace(-1, 1, nu)
    detector_v = np.linspace(-1, 1, nv)
    
    # Create angle grid
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    
    # Pre-allocate lists for CSR matrix construction
    rows, cols, data = [], [], []
    
    # Build the matrix row by row
    for ia, theta in enumerate(angles):
        # Calculate ray direction and detector plane basis vectors
        d = np.array([np.cos(theta), np.sin(theta), 0.0])  # parallel beam direction
        du_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])  # u axis on detector
        dv_hat = np.array([0.0, 0.0, 1.0])  # v axis (z)
        
        # For each detector position
        for iu, u in enumerate(detector_u):
            for iv, v in enumerate(detector_v):
                row_idx = ia * nu * nv + iu * nv + iv
                
                # Calculate ray offset in detector plane
                ray_offset = u * du_hat + v * dv_hat
                
                # Calculate signed distance along ray and perpendicular distance
                s = np.dot(vox_centers - ray_offset, d)
                dist2 = np.sum((vox_centers - ray_offset - np.outer(s, d)) ** 2, axis=1)
                
                # Use Gaussian kernel to approximate the line integral
                sigma = 1.0 / min(nx, ny)  # Adjust this parameter to control the width of the ray
                weights = np.exp(-0.5 * dist2 / (sigma ** 2))
                weights = weights / np.sum(weights)  # Normalize
                
                # Store non-zero weights in CSR format
                nonzero_idx = np.where(weights > 1e-10)[0]
                rows.extend([row_idx] * len(nonzero_idx))
                cols.extend(nonzero_idx)
                data.extend(weights[nonzero_idx])
    
    # Create sparse matrix
    A = csr_matrix((data, (rows, cols)), 
                   shape=(n_angles * nu * nv, nx * ny * nz))
    
    return A

def apply_radon_matrix_3d(
    A: csr_matrix,
    volume: np.ndarray,
    volume_shape: Tuple[int, int, int],
    n_angles: int,
    det_shape: Tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Apply the 3D Radon transform matrix to a volume.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The 3D Radon transform matrix
    volume : np.ndarray
        Input volume of shape (nx, ny, nz)
    volume_shape : Tuple[int, int, int]
        Shape of the volume (nx, ny, nz)
    n_angles : int
        Number of projection angles
    det_shape : Tuple[int, int], optional
        Shape of the detector (nu, nv)
        
    Returns
    -------
    np.ndarray
        Sinogram of shape (nu, nv, n_angles)
    """
    nx, ny, nz = volume_shape
    if det_shape is None:
        nu = int(np.ceil(np.sqrt(2) * nx))
        nv = int(np.ceil(np.sqrt(2) * ny))
    else:
        nu, nv = det_shape
    
    # Flatten the volume
    x = volume.flatten()
    
    # Apply the matrix
    sino_flat = A @ x
    
    # Reshape to sinogram format
    return sino_flat.reshape(n_angles, nu, nv).transpose(1, 2, 0)

def demo_radon_matrix_3d():
    """
    Demonstrate the 3D Radon transform matrix construction and application.
    """
    # Parameters
    volume_shape = (15, 15, 15)
    n_angles = 180
    det_shape = (5, 5)
    
    # Build the matrix
    A = build_radon_matrix_3d(volume_shape, n_angles, det_shape)
    
    # Create a test volume (a simple sphere)
    x = np.linspace(-1, 1, volume_shape[0])
    y = np.linspace(-1, 1, volume_shape[1])
    z = np.linspace(-1, 1, volume_shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    test_volume = (X**2 + Y**2 + Z**2 < 0.5).astype(float)
    
    # Apply the transform
    sinogram = apply_radon_matrix_3d(A, test_volume, volume_shape, n_angles, det_shape)
    
    # Visualize middle slice of volume and corresponding sinogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    mid_slice = volume_shape[2] // 2
    ax1.imshow(test_volume[:, :, mid_slice], cmap='gray')
    ax1.set_title('Middle Slice of Test Volume')
    ax1.axis('off')
    
    mid_angle = n_angles // 2
    ax2.imshow(sinogram[:, :, mid_angle], cmap='gray')
    ax2.set_title('Middle Angle Projection')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
# Run the demos
# demo_radon_matrix()
demo_radon_matrix_3d()

# Plot SVD spectra for different image sizes
plot_svd_spectra(do_3d=True)

# %%
