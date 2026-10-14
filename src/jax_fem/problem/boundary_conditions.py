"""
Dirichlet boundary condition data.
"""

from __future__ import annotations

import dataclasses

from jax_fem.mesh.dfn import DfnMesh
from jax_fem.mesh.triangle import TriangleMesh
from jax_fem.problem.elliptic import ScalarCoefficient


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class DirichletCondition:
    """
    A Dirichlet value function on a set of mesh boundary tags.

    Attributes
    ----------
    boundary_tags : tuple[int, ...]
        Mesh boundary facet tags the condition applies to.
    value : ScalarCoefficient
        Dirichlet value function ``g(x)``.

    Notes
    -----
    The mesh decides which facets carry each tag, the finite element space
    maps those facets to DOFs, and this class selects the Dirichlet tags and
    their value function.
    """

    boundary_tags: tuple[int, ...]
    value: ScalarCoefficient


def create_full_boundary_dirichlet_condition(
    mesh: TriangleMesh | DfnMesh, value: ScalarCoefficient
) -> DirichletCondition:
    """
    Dirichlet condition on the whole mesh boundary.

    Parameters
    ----------
    mesh : TriangleMesh | DfnMesh
        Mesh whose boundary tags are all covered.
    value : ScalarCoefficient
        Dirichlet value function ``g(x)``.

    Returns
    -------
    condition : DirichletCondition
        Condition on every tag in ``mesh.boundary_tag_names``.
    """
    return DirichletCondition(
        boundary_tags=tuple(mesh.boundary_tag_names.values()), value=value
    )
