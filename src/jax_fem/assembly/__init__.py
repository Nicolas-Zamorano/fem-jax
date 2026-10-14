"""
Cellwise integration and global assembly.
"""

from jax_fem.assembly.global_assembly import (
    GlobalMatrix,
    assemble_matrix,
    assemble_vector,
    combine_matrix_blocks,
    gather_vector,
)
from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.assembly.mixed import assemble_mixed_system

__all__ = [
    "integrate_cellwise",
    "GlobalMatrix",
    "assemble_matrix",
    "assemble_vector",
    "gather_vector",
    "combine_matrix_blocks",
    "assemble_mixed_system",
]
