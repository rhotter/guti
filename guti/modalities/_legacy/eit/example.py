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
from head_mesh import get_sphere

imaging_method = 'eit'

# %%
# Specify tissue-related properties.

# layers from outside in: scalp, skull, csf, grey matter, white matter
outer_radii = [7, 6.5, 6, 5.8, 5.5]  # [cm]
conductivities = [3.3e-3, 4.2e-5, 1.79e-3, 0.33e-3, 0.14e-3]  # [S/cm]
# conductivities = [0.2e-3, 0.2e-3, 0.2e-3, 0.2e-3, 0.2e-3]

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
# Define electrode pair
num_electrodes = 2
electrode_positions = np.array([
    [0.0, 0.0, outer_radii[0]],     # Top electrode
    [outer_radii[0], 0.0, 0.0]      # Right electrode (90 degrees from top)
])

# Define single injection pattern for the electrode pair
injection_patterns = np.array([[1.0, -1.0]])  # One pattern: inject at first, sink at second
injection_current = 1e-3  # [A] - injection current magnitude

# %%
# Define the problem.

class EIT(Problem):
    """
    We solve the problem:
        ∇·(σ ∇φ) = 0 in Ω,
        φ = V on Γ_D,
        σ∇φ ⋅ n = J on Γ_N.
    where:
        - Ω is the domain of interest,
        - Γ_D is the Dirichlet boundary where voltage is applied,
        - Γ_N is the Neumann boundary where current is applied,
        - φ is the electric potential,
        - σ is the electrical conductivity distribution,
        - V is the voltage,
        - J is the current.

    ----------------------------------------------------------------
    Generally, JAX-FEM solves -∇·(f(∇u)) = b. 
    Therefore, f(∇u) = - σ∇φ and b = 0.

    Note: In EIT, we typically:
    1. Apply known voltage patterns or current patterns at boundary
    2. Measure the resulting currents or voltages at other electrodes
    3. Use these measurements to reconstruct the internal conductivity distribution σ
    """

    def set_params(self, params):
        sigma = params[0]
        self.sigma = sigma

    def custom_init(self, *args):
        self.fe = self.fes[0]
        quad_pts = self.fe.get_physical_quad_points() # (num_cells, num_quads, dim)
        self.internal_vars = (quad_pts, )
    
    # Define the function f with variable conductivity.
    def get_tensor_map(self):
        def tensor_map(u_grad, x, *args):
            quad_pts = self.internal_vars[0]
            matches = np.all(quad_pts == x[None, None, :], axis=-1)  # (num_cells, num_quads)
            sigma_x = np.sum(self.sigma * matches)  # scalar value for this x
            return -sigma_x * u_grad
        return tensor_map
    
    # Define the source term b (no internal sources in EIT)
    def get_mass_map(self):
        def mass_map(u, x, *args):
            return np.array([0.])
        return mass_map

    # Define surface flux functions (Neumann BCs)
    def get_surface_maps(self):

        def surface_map(u, x):
            # Apply current pattern using Gaussian distribution for smoother injection
            percent_peak = 0.01  # percent of nearest neighbor value relative to peak
            std = np.sqrt(-(mesh_size**2)/(2*np.log(percent_peak)))
            
            # Calculate distances to all electrodes
            distances = np.array([np.linalg.norm(x - pos) for pos in electrode_positions])
            
            # Apply Gaussian distribution for each electrode
            currents = injection_patterns[0] * np.exp(-np.power(distances, 2) / (2 * std**2))
            total_current = np.sum(currents) * injection_current
            
            return np.array([total_current])


        # def surface_map(u, x):
        #     # Calculate normal vector at surface point
        #     normal = x / np.linalg.norm(x)  # For sphere, normal is radial direction
            
        #     # For each electrode, determine if this surface point is in the electrode area
        #     electrode_currents = []
        #     for pos, pattern in zip(electrode_positions, injection_patterns[0]):
        #         # Project surface point onto unit vector in electrode direction
        #         electrode_dir = pos / np.linalg.norm(pos)
        #         projection = np.dot(normal, electrode_dir)
                
        #         # Define electrode area using angle from electrode direction
        #         angle_threshold = np.pi/8  # 22.5 degrees
        #         is_in_electrode = projection > np.cos(angle_threshold)
                
        #         # Scale current by area factor (using JAX operations)
        #         area_factor = 1.0 / (2.0 * np.pi * (1.0 - np.cos(angle_threshold)))
        #         current = pattern * injection_current * np.where(is_in_electrode, area_factor, 0.0)
        #         electrode_currents.append(current)
            
        #     # Sum contributions from all electrodes
        #     total_current = np.sum(np.array(electrode_currents))
        #     return np.array([total_current])
            
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
# Create problem instance
problem = EIT(
    mesh=mesh,
    vec=1,
    dim=3,
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=neumann_location_fns
)

# Specify parameters to optimize over
fe = problem.fes[0]
quad_pts = fe.get_physical_quad_points() # (num_cells, num_quads, dim)
radii = np.linalg.norm(quad_pts, axis=-1)
sigma = np.zeros((fe.num_cells, fe.num_quads))
sigma = np.where(radii > outer_radii[0], conductivities[0],
        np.where(radii > outer_radii[1], conductivities[1],
        np.where(radii > outer_radii[2], conductivities[2],
        np.where(radii > outer_radii[3], conductivities[3], conductivities[4]))))
params = [sigma]

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
# save_sol(problem.fe, sol_list[0], vtk_path, cell_infos=[('dsource', dsource[0])])

# %%
# Compute Jacobian

# output_size = forward_op(params).shape[0]
# input_size = params[0].size
# print(f"Output size: {output_size}, Input size: {input_size}")
J = jax.jacrev(forward_op)(params)[0][..., 0]
print(J.shape)
save_sol(problem.fe, sol_list[0], vtk_path, cell_infos=[('jac0', J[0,0])])

# %%
# Compute SVD of Jacobian
s = np.linalg.svdvals(J.reshape(J.shape[0], -1))

# Plot the singular values.
plt.figure(figsize=(8, 5))
plt.semilogy(s)
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
