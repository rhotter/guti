import jax
import jax.numpy as jnp

def compute_barycentrics(a, b, c, d, p, eps=1e-12):
    T = jnp.column_stack((b - a, c - a, d - a))
    rhs = p - a
    cond = jnp.linalg.cond(T)
    uvw  = jnp.where(
        cond < 1.0 / eps,
        jnp.linalg.solve(T, rhs),
        jnp.full((3,), jnp.nan),
    )
    return jnp.concatenate((jnp.array([1.0]) - uvw.sum(keepdims=True), uvw))

def find_containing_tetrahedron(mesh, x0, tol=1e-8):
    for cell in mesh.cells:
        verts = mesh.points[cell]
        bary = compute_barycentrics(*verts, x0)
        inside = (
            jnp.all(jnp.isfinite(bary))
            & jnp.all(bary >= -tol)
            & jnp.all(bary <= 1.0 + tol)
        )
        if inside:
            return cell, bary
    raise ValueError("Point x0 is not inside any tetrahedron in the mesh.")

def get_neighbor_indices(mesh, node_idx):
    """Return *1‑ring* neighbours of *node_idx* (ascending order, excl. itself)."""
    mask   = jnp.any(mesh.cells == node_idx, axis=1)  # (M,)
    neighs = jnp.unique(mesh.cells[mask])
    return neighs[neighs != node_idx]

def compute_venant_loads(mesh, x0, p, lam=1e-6, aref=None, tol=1e-8, include_central=True):
    
    # ‑‑ Locate tetrahedron and choose its vertex closest to the dipole centre.
    tet_nodes, _   = find_containing_tetrahedron(mesh, x0, tol)
    pos_tet        = mesh.points[tet_nodes]
    distances      = jnp.linalg.norm(pos_tet - x0, axis=1)
    v_center       = tet_nodes[jnp.argmin(distances)]

    # ‑‑ Gather neighbourhood and optionally prepend the central vertex.
    neigh = get_neighbor_indices(mesh, v_center)
    if include_central:
        neigh = jnp.concatenate((jnp.array([v_center]), neigh))
    pos_neigh = mesh.points[neigh]

    # ‑‑ Reference length for scaling (mean distance to *x0*).
    if aref is None:
        aref = jnp.mean(jnp.linalg.norm(pos_neigh - x0, axis=1))
    if aref <= 0.0:
        raise ValueError("Reference length must be positive.")
    
    # ‑‑ Build scaled moment matrix.
    diff = (pos_neigh - x0) / aref                         # (K,3)
    N    = diff.shape[0]
    Xbar = jnp.vstack((jnp.ones((1, N)), diff.T))          # (4,K)
    tbar = jnp.concatenate((jnp.zeros((1,)), p / aref))    # (4,)

    # ‑‑ Solve regularised least‑squares  (XᵀX + λI) q = Xᵀ t.
    M   = Xbar.T @ Xbar + lam * jnp.eye(N)
    rhs = Xbar.T @ tbar
    cond = jnp.linalg.cond(M)
    q    = jnp.where(cond < 1e12, jnp.linalg.solve(M, rhs), jnp.zeros(N))

    # Force‑balance safeguard if centre vertex was excluded (Σq = 0).
    if not include_central:
        q -= q.sum() / N

    # ‑‑ Scatter into global load vector.
    load = jnp.zeros((mesh.points.shape[0],), dtype=jnp.float32)
    load = load.at[neigh].set(q.astype(jnp.float32))
    
    return neigh, load, mesh.points
