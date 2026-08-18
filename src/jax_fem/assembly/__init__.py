"""
Local integration and global matrix/vector assembly.
"""

from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.assembly.matrix import GlobalMatrix, assemble_bilinear_form
from jax_fem.assembly.vector import assemble_linear_form

__all__ = [
    "integrate_cellwise",
    "GlobalMatrix",
    "assemble_bilinear_form",
    "assemble_linear_form",
]
