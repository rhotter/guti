# %%
# Common imports.
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import os
import jax
import jax.numpy as np
print(f"JAX device: {jax.devices()}")

import matplotlib.pyplot as plt
import meshio
import gmsh
import sys
sys.path.append('..')
sys.path.append('../..')

from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh
from head_mesh import get_sphere, get_multilayer_sphere
from eeg_sources import compute_venant_loads

imaging_method = 'eeg'

# %%
# Specify tissue-related properties.

# layers from outside in: scalp, skull, csf, grey matter, white matter
outer_radii = [7, 6.5, 6, 5.8, 5.5]  # [cm]
conductivities = [3.3e-3, 4.2e-5, 1.79e-3, 0.33e-3, 0.14e-3]  # [S/cm]
# conductivities = [0.2e-3, 0.2e-3, 0.2e-3, 0.2e-3, 0.2e-3]  # [S/cm]

# %%
# Specify dipole source-related information.
dipole_center = np.array([0.0, 0.0, 5.6]) # [cm]
dipole_moment = np.array([0.0, 0.0, 1e-5]) # [A·cm]

# %%
# Specify mesh-related information. 
ele_type = 'TET4'
mesh_size = outer_radii[0] / 11  # [cm]
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)
meshio_mesh = get_sphere(outer_radii[0], mesh_size, data_dir, ele_type=ele_type)
# meshio_mesh = get_multilayer_sphere(outer_radii, mesh_size, data_dir, ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[get_meshio_cell_type(ele_type)])

def sphere_outer_boundary(point):
    tol = mesh_size * 1e-3
    mask = np.abs(np.linalg.norm(point) - outer_radii[0]) < tol
    return mask

def sphere_reference_boundary(point):
    tol = mesh_size * 1e-3
    reference_position = np.array([0.0, 0.0, -outer_radii[0]])
    return np.linalg.norm(point - reference_position) < tol

# %%
# get dipole loads using the Venant approach.
aref = mesh_size / 11 # [cm]
neighbor_indices, load_values, load_node_positions = compute_venant_loads(
    mesh, dipole_center, dipole_moment, aref=aref)
print(f"load_values with precision: {[f'{val:.2e}' for val in load_values]}")

# %%
# Define the problem.

class EEG(Problem):
    """
    We solve the problem:
        ∇·(σ ∇φ) = - ∇·J_p in Ω,
        φ = 0 on Γ_D,
        σ∇φ ⋅ n = 0 on Γ_N.
    where:
        - Ω is the domain of interest,
        - Γ_D is the Dirichlet boundary of the domain,
        - Γ_N is the Neumann boundary of the domain,
        - φ is the electric potential,
        - σ is the electric conductivity, 
        - J_p is the primary current density.

    ----------------------------------------------------------------
    Generally, JAX-FEM solves -∇·(f(∇u)) = b. 
    Therefore, f(∇u) = - σ∇φ and b = - ∇·J_p.
    """

    def set_params(self, params):
        load_values = params[0]
        self.load_values = load_values

    def custom_init(self, *args):
        self.fe = self.fes[0]
        quad_pts = self.fe.get_physical_quad_points() # (num_cells, num_quads, dim)
        radii = np.linalg.norm(quad_pts, axis=-1)
        sigma = np.zeros((self.fe.num_cells, self.fe.num_quads))
        sigma = np.where(radii > outer_radii[0], conductivities[0],
                np.where(radii > outer_radii[1], conductivities[1],
                np.where(radii > outer_radii[2], conductivities[2],
                np.where(radii > outer_radii[3], conductivities[3], conductivities[4]))))
        self.internal_vars = (quad_pts, sigma)

    # Define the function f with variable conductivity.
    def get_tensor_map(self):
        
        def tensor_map(u_grad, x, *args):
            quad_pts = self.internal_vars[0]
            sigma = self.internal_vars[1]
            matches = np.all(quad_pts == x[None, None, :], axis=-1)  # (num_cells, num_quads)
            sigma_x = np.sum(sigma * matches)  # scalar value for this x
            return -sigma_x * u_grad

        # def tensor_map(u_grad, x, *args):
        #     return -self.sigma * u_grad
        
        return tensor_map
    
    # Define the source term b
    def get_mass_map(self):

        # def mass_map(u, x, *args):
        #     val = -np.array([10*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
        #     return val

        # def mass_map(u, x, *args):
        #     dipole_center = np.array([6.5, 0.0, 0.0])
        #     std = 0.02
        #     amplitude = 1e4 # [nA]
        #     r = np.linalg.norm(x - dipole_center)
        #     val = -np.array([amplitude * np.exp(-(np.power(r, 2)) / std)])
        #     return val
        
        # def mass_map(u, x, *args):
        #     percent_peak = 0.01 # percent of nearest neighbor value relative to peak
        #     std = np.sqrt(-(mesh_size**2)/(2*np.log(percent_peak)))
        #     normalization_factor = (2 * np.pi * std**2)**(3/2)
        #     s = 0
        #     for xi, qi in zip(self.load_node_positions, self.load_values):
        #         r = np.linalg.norm(x - xi)
        #         s += qi * np.exp(-np.power(r, 2) / (2 * std**2)) / normalization_factor
        #     return np.array([s])

        def mass_map(u, x, *args):
            percent_peak = 0.01 # percent of nearest neighbor value relative to peak
            std = np.sqrt(-(mesh_size**2)/(2*np.log(percent_peak)))
            normalization_factor = (2 * np.pi * std**2)**(3/2)
            r = np.linalg.norm(x - load_node_positions, axis=1)
            s = np.sum(self.load_values * np.exp(-np.power(r, 2) / (2 * std**2)) / normalization_factor)
            return np.array([s])
        
        return mass_map

    # Define surface flux functions (Neumann BCs)
    def get_surface_maps(self):
        
        # def surface_map(u, x):
        #     quad_pts = self.internal_vars[0]
        #     matches = np.all(quad_pts == x[None, None, :], axis=-1)  # (num_cells, num_quads)
        #     sigma_x = np.sum(self.sigma * matches)  # scalar value for this x
        #     return -np.array([0 / sigma_x])
        
        def surface_map(u, x):
            return -np.array([0])
        
        return [surface_map]

# %%
# Define boundary conditions and locations.
def dirichlet_val(point):
    return 0.

dirichlet_location_fns = [sphere_reference_boundary]
vecs = [0]
value_fns = [dirichlet_val]
dirichlet_bc_info = [dirichlet_location_fns, vecs, value_fns]
# dirichlet_bc_info = None

neumann_location_fns = [sphere_outer_boundary]

# %%
# Create an instance of the problem (3D, scalar field) and solve.
problem = EEG(
    mesh=mesh,
    vec=1,           # scalar problem
    dim=3,           # 3D problem
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=neumann_location_fns
)

# Specify parameters to optimize over
params = [load_values]

# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem)
sol_list = fwd_pred(params)
# sol_list = solver(problem)

# Save the solution
vtk_dir = os.path.join(data_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)
vtk_path = os.path.join(vtk_dir, f'{imaging_method}.vtu')
save_sol(problem.fe, sol_list[0], vtk_path)


# %%
# Compute gradient of scalar-valued objective function

# all sensors on surface
def get_sensor_mask(points):
    tol = mesh_size * 1e-3
    radii = np.linalg.norm(points, axis=1)
    sensor_mask = np.abs(radii - outer_radii[0]) < tol
    return sensor_mask

# # single sensor at surface
# def get_sensor_mask(points):
#     tol = mesh_size * 1e-3
#     sensor_position = np.array([0.0, 0.0, outer_radii[0]])
#     sensor_mask = np.all(np.abs(points - sensor_position) < tol, axis=1)
#     return sensor_mask

def get_sensor_recordings(sol):
    mesh_points = problem.fe.mesh.points
    surface_mask = get_sensor_mask(mesh_points)
    sensor_recordings = sol[surface_mask]
    return sensor_recordings

def forward_op(params):
    electric_potential = fwd_pred(params)[0]
    sensor_recordings = get_sensor_recordings(electric_potential)
    return sensor_recordings

def objective_fn(params):
    sensor_recordings = forward_op(params)
    return np.linalg.norm(sensor_recordings) ** 2

# dsource = jax.grad(objective_fn)(params)
# print(dsource[0].shape)
# save_sol(problem.fe, sol_list[0], vtk_path, point_infos=[('dsource', dsource[0])])

# %%
# Compute Jacobian

# output_size = forward_op(params).shape[0]
# input_size = params[0].size
# print(f"Output size: {output_size}, Input size: {input_size}")
J = jax.jacrev(forward_op)(params)[0]
print(J.shape)
save_sol(problem.fe, sol_list[0], vtk_path, point_infos=[('jac0', J[0,0])])

# %%
# Compute SVD of Jacobian
s = np.linalg.svdvals(J.reshape(J.shape[0], -1))

# Plot the singular values.
plt.figure(figsize=(8, 5))
plt.semilogy(s, 'o-', linewidth=2, markersize=5)
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value (log scale)')
plt.title('Singular Values of the Jacobian')
plt.grid(True)
plot_path = os.path.join(data_dir, f'{imaging_method}_singular_values.png')
plt.savefig(plot_path)
plt.show()


# %%
from data_utils import save_svd
save_svd(s, imaging_method)
# %%