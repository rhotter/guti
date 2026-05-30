import numpy as np
import matplotlib.pyplot as plt

from guti.core import get_sensor_positions, get_grid_positions, BRAIN_RADIUS

SPHERE_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])


def sarvas_before(r, r0):
    """
    Pre-fix Sarvas implementation from guti/modalities/meg/meg.py.
    Uses absolute coordinates (mm) and the old matrix form.
    """
    mu0 = 4 * np.pi * 1e-7
    r = np.asarray(r)
    r0 = np.asarray(r0)
    R = r - r0
    a = np.linalg.norm(R)
    r_norm = np.linalg.norm(r)
    if a < 1e-12 or r_norm < 1e-12:
        return np.zeros((3, 3))
    F = a * (a * r_norm + r_norm**2 - np.dot(r0, r))
    nabla_F = (
        (a**2 / r_norm + np.dot(R, r) / a + 2 * a + 2 * r_norm) * r
        - (a + 2 * r_norm + np.dot(R, r) / a) * r0
    )
    R_cross = np.array(
        [
            [0.0, -R[2], R[1]],
            [R[2], 0.0, -R[0]],
            [-R[1], R[0], 0.0],
        ]
    )
    M = (mu0 / (4 * np.pi)) * (F - np.dot(r, nabla_F)) / (F**2) * R_cross
    return M


def sarvas_after(r, r0, center=SPHERE_CENTER):
    """
    Corrected Sarvas implementation (spherical conductor), with
    centered coordinates and mm -> m conversion.
    """
    mu0 = 4 * np.pi * 1e-7
    r = (np.asarray(r) - center) * 1e-3
    r0 = (np.asarray(r0) - center) * 1e-3
    a_vec = r - r0
    a = np.linalg.norm(a_vec)
    r_norm = np.linalg.norm(r)
    if a < 1e-12 or r_norm < 1e-12:
        return np.zeros((3, 3))
    F = a * (a * r_norm + r_norm**2 - np.dot(r0, r))
    if abs(F) < 1e-20:
        return np.zeros((3, 3))
    a_dot_r = np.dot(a_vec, r)
    nabla_F = (
        (a**2 / r_norm + a_dot_r / a + 2 * a + 2 * r_norm) * r
        - (a + 2 * r_norm + a_dot_r / a) * r0
    )
    r0_cross = np.array(
        [
            [0.0, -r0[2], r0[1]],
            [r0[2], 0.0, -r0[0]],
            [-r0[1], r0[0], 0.0],
        ]
    )
    r0xr = r0_cross @ r
    M = (mu0 / (4 * np.pi)) * (-F * r0_cross - np.outer(nabla_F, r0xr)) / (F**2)
    return M


def compute_svd(n_sensors, grid_spacing_mm, offset_mm, leadfield_fn):
    sensors = get_sensor_positions(n_sensors, offset=offset_mm)
    sources = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    n_sources = len(sources)
    A = np.zeros((3 * n_sensors, 3 * n_sources))
    for i, sensor in enumerate(sensors):
        for j, source in enumerate(sources):
            M = leadfield_fn(sensor, source)
            A[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = M
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    return s


def plot_before_after(n_sensors=400, grid_spacing_mm=15.0, offsets_mm=(7, 25)):
    labels = {7: "OPM", 25: "SQUID"}
    plt.figure(figsize=(10, 6))
    for offset in offsets_mm:
        s_before = compute_svd(n_sensors, grid_spacing_mm, offset, sarvas_before)
        s_after = compute_svd(n_sensors, grid_spacing_mm, offset, sarvas_after)
        plt.semilogy(
            s_before,
            label=f"{labels.get(offset, offset)} before",
            linestyle="--",
        )
        plt.semilogy(
            s_after,
            label=f"{labels.get(offset, offset)} after",
        )
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value")
    plt.title(f"MEG SVD spectra (n_sensors={n_sensors}, spacing={grid_spacing_mm} mm)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.legend()
    out_path = "plots/meg_sarvas_before_after_400s_15mm.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    plot_before_after()
