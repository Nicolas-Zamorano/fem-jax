"""Mesh geometry, topology, tags, and adaptive refinement."""

from jax_fem.mesh.amr import (
    adapt_mesh_from_error_estimator,
    compute_target_cell_sizes,
    mark_cells_by_dorfler_bulk_criterion,
    refine_mesh_longest_edge_bisection,
)
from jax_fem.mesh.base import Mesh
from jax_fem.mesh.dfn import (
    DfnMesh,
    create_dfn_mesh_from_arrays,
    create_dfn_two_fractures_mesh,
    create_fracture_frames,
    read_dfn_mesh_avs,
)
from jax_fem.mesh.gmsh import (
    create_gmsh_l_shaped_mesh,
    create_gmsh_mesh_hierarchy,
    create_gmsh_unit_square_mesh,
    create_triangle_mesh_from_gmsh_model,
    l_shaped_gmsh_geometry,
    read_triangle_mesh_from_msh,
    run_in_owned_gmsh_session,
    unit_square_gmsh_geometry,
)
from jax_fem.mesh.triangle import (
    BoundaryData,
    TriangleMesh,
    cell_diameters,
    create_triangle_mesh_from_arrays,
)

__all__ = [
    "Mesh",
    "TriangleMesh",
    "BoundaryData",
    "cell_diameters",
    "create_triangle_mesh_from_arrays",
    "create_gmsh_unit_square_mesh",
    "create_gmsh_l_shaped_mesh",
    "create_gmsh_mesh_hierarchy",
    "unit_square_gmsh_geometry",
    "l_shaped_gmsh_geometry",
    "run_in_owned_gmsh_session",
    "create_triangle_mesh_from_gmsh_model",
    "read_triangle_mesh_from_msh",
    "DfnMesh",
    "create_dfn_mesh_from_arrays",
    "read_dfn_mesh_avs",
    "create_dfn_two_fractures_mesh",
    "create_fracture_frames",
    "mark_cells_by_dorfler_bulk_criterion",
    "compute_target_cell_sizes",
    "refine_mesh_longest_edge_bisection",
    "adapt_mesh_from_error_estimator",
]
