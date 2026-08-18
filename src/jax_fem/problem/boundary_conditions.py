"""Dirichlet boundary condition data.

See Section 12.2 of the architecture specification.
"""

from __future__ import annotations

import dataclasses

from jax_fem.mesh.triangle import TriangleMesh
from jax_fem.problem.elliptic import ScalarCoefficient


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class DirichletCondition:
    """Association between a set of mesh boundary tags and a Dirichlet value.

    Boundary tags belong to the mesh; the boundary function and the
    selection of tags belong to the problem (Section 12.2). The mesh
    determines which facets carry each tag; the finite element space maps
    those facets to DOFs; this class specifies which tags are Dirichlet and
    which value function applies.

    Attributes
    ----------
    boundary_tags:
        Mesh boundary facet tags this condition applies to.
    value:
        ``g(x)``, the Dirichlet value function.
    """

    boundary_tags: tuple[int, ...]
    value: ScalarCoefficient


def create_full_boundary_dirichlet_condition(
    mesh: TriangleMesh, value: ScalarCoefficient
) -> DirichletCondition:
    """Convenience constructor: one Dirichlet value on the entire mesh boundary.

    Applies ``value`` to every boundary tag the mesh currently has, i.e.
    every tag in ``mesh.boundary_tag_names`` (Section 12.2).

    Parameters
    ----------
    mesh:
        The mesh whose boundary tags to cover.
    value:
        ``g(x)``, the Dirichlet value function.

    Returns
    -------
    DirichletCondition
    """
    return DirichletCondition(
        boundary_tags=tuple(mesh.boundary_tag_names.values()), value=value
    )
