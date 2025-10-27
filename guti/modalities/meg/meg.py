# %%
%load_ext autoreload
%autoreload 2

# %%
opm_distance = 7   # mm
squid_distance = 25

# %%
import numpy as np
from guti.core import get_sensor_positions, get_grid_positions

def sarvas_formula(r, r0):
    """
    Compute the 3×3 lead‐field matrix M such that B = M @ q,
    using the Sarvas formula for a spherical conductor.

    Parameters
    ----------
    r : array-like, shape (3,)
        Sensor position in head coordinates.
    r0 : array-like, shape (3,)
        Dipole position in head coordinates.

    Returns
    -------
    M : ndarray, shape (3, 3)
        Lead‐field matrix.
    """
    mu0 = 4*np.pi*1e-7           # vacuum permeability
    R = np.asarray(r) - np.asarray(r0)
    a = np.linalg.norm(R)
    r_norm = np.linalg.norm(r)

    F = a * (a * r_norm + r_norm ** 2 - np.dot(r0, r))
    nabla_F = (a**2 / r_norm + np.dot(R, r) / a + 2 * a + 2 * r_norm) * r - (a + 2 * r_norm + np.dot(R, r) / a) * r0

    # Cross‐product matrix for R
    R_cross = np.array([
        [    0, -R[2],  R[1]],
        [ R[2],     0, -R[0]],
        [-R[1],  R[0],     0]
    ])

    # Lead‐field matrix
    M = (mu0/(4*np.pi)) * (F - np.dot(r, nabla_F)) / (F**2) * R_cross

    return M

def compute_forward_matrix(n_sensors=4000, grid_spacing_mm=5.0, offset=opm_distance):
    """
    Compute the MEG forward matrix using Sarvas's formula.
    
    Parameters
    ----------
    n_sensors : int
        Number of sensors to use
    grid_spacing_mm : float
        Spacing between grid points in mm
    offset : float
        Offset from the origin in mm
        
    Returns
    -------
    A : ndarray, shape (3*n_sensors, 3*n_sources)
        Forward matrix that maps dipole moments to sensor measurements
    """
    # Get sensor and source positions using core.py functions
    sensors = get_sensor_positions(n_sensors, offset=offset)
    sources = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    n_sources = len(sources)
    
    # Initialize forward matrix
    A = np.zeros((3*n_sensors, 3*n_sources))
    print(f"The number of sensors is {n_sensors} and the number of sources is {n_sources}, the shape of the forward matrix is {A.shape}")
    
    # Compute lead field matrix for each sensor-source pair
    for i, sensor in enumerate(sensors):
        for j, source in enumerate(sources):
            # Compute lead field matrix
            M = sarvas_formula(sensor, source)
            
            # Fill in the corresponding 3x3 block in the forward matrix
            A[3*i:3*(i+1), 3*j:3*(j+1)] = M
            
    return A

def compute_svd(n_sensors=4000, grid_spacing_mm=5.0, offset=opm_distance):
    """
    Compute the singular value decomposition of the MEG forward matrix.
    
    Parameters
    ----------
    n_sensors : int
        Number of sensors to use
    grid_spacing_mm : float
        Spacing between grid points in mm
        
    Returns
    -------
    s : ndarray
        Singular values of the forward matrix
    """
    A = compute_forward_matrix(n_sensors, grid_spacing_mm, offset)
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    return A, s


# %%
%matplotlib inline
from matplotlib import pyplot as plt

# A_1, s_1 = compute_svd(n_sensors=200, n_sources=1000, offset=opm_distance)
# A_2, s_2 = compute_svd(n_sensors=200, n_sources=1000, offset=squid_distance)

n_sensors = 1000
grid_spacing_mm = 5.0

A_1, s_1 = compute_svd(n_sensors=n_sensors, grid_spacing_mm=grid_spacing_mm, offset=opm_distance)
A_2, s_2 = compute_svd(n_sensors=n_sensors, grid_spacing_mm=grid_spacing_mm, offset=squid_distance)

# Plot the singular value spectra on a log scale
plt.figure(figsize=(8, 4))
plt.semilogy(s_1, label='OPM')
plt.semilogy(s_2, label='SQUID')
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value (log scale)')
plt.title(f'Singular Value Spectra ({n_sensors} sensors, {grid_spacing_mm} mm grid spacing)')
plt.grid(True)
plt.legend()
plt.show()

# %%
from guti.parameters import Parameters
from guti.data_utils import save_svd
for n_sensors in [50, 100, 200, 500, 700]:
    for grid_spacing_mm in [5.0, 10.0, 15, 20.0, 30]:
        A, s = compute_svd(n_sensors=n_sensors, grid_spacing_mm=grid_spacing_mm, offset=opm_distance)
        save_svd(s, f'meg_opm', Parameters(num_sensors=n_sensors, source_spacing_mm=grid_spacing_mm, sensor_offset_mm=opm_distance))
        print(f'Saved SVD for {n_sensors} sensors, {grid_spacing_mm} mm grid spacing')
        A, s = compute_svd(n_sensors=n_sensors, grid_spacing_mm=grid_spacing_mm, offset=squid_distance)
        save_svd(s, f'meg_squid', Parameters(num_sensors=n_sensors, source_spacing_mm=grid_spacing_mm, sensor_offset_mm=squid_distance))
        print(f'Saved SVD for {n_sensors} sensors, {grid_spacing_mm} mm grid spacing')


# %%
import sys

sys.path.append('../..')
from guti.data_utils import save_svd

save_svd(s_1, 'meg_opm')
save_svd(s_2, 'meg_squid')

def plot_svd_spectra(n_sensors_list=[50, 100, 200, 500, 700, 1000], n_sources=1000):
    """
    Compute and plot SVD spectra for different numbers of sensors.
    
    Parameters
    ----------
    n_sensors_list : list of int
        List of different numbers of sensors to compare
    n_sources : int
        Number of source dipoles to use
    """
    plt.figure(figsize=(10, 6))
    
    for n_sensors in n_sensors_list:
        # Compute forward matrix and SVD
        A = compute_forward_matrix(n_sensors, n_sources)
        _, s, _ = np.linalg.svd(A, full_matrices=False)
        
        # Plot singular values
        plt.semilogy(s, label=f'{n_sensors} sensors')
    
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('MEG Singular Value Spectra for Different Numbers of Sensors')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_svd_spectra()



# %%
