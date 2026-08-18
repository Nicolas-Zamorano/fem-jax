"""Mesh geometry, topology, and tags (Section 7)."""

from jax_fem.mesh.base import Mesh
from jax_fem.mesh.gmsh import (
    create_triangle_mesh_from_gmsh_model,
    read_triangle_mesh_from_msh,
)
from jax_fem.mesh.triangle import (
    BoundaryData,
    TriangleMesh,
    create_structured_unit_square_mesh,
    create_triangle_mesh_from_arrays,
)

__all__ = [
    "Mesh",
    "TriangleMesh",
    "BoundaryData",
    "create_triangle_mesh_from_arrays",
    "create_structured_unit_square_mesh",
    "create_triangle_mesh_from_gmsh_model",
    "read_triangle_mesh_from_msh",
]
