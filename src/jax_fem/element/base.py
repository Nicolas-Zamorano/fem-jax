"""
FiniteElement interface shared by all finite elements.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax

from jax_fem.reference_cell.base import ReferenceCell


@runtime_checkable
class FiniteElement(Protocol):
    """
    Local basis and degrees of freedom on a reference cell.

    Attributes
    ----------
    reference_cell : ReferenceCell
        The reference cell the element is defined on.
    polynomial_degree : int
        Polynomial degree of the local space.
    number_of_local_dofs : int
        Number of local basis functions, ``N_test``.
    value_shape : tuple[int, ...]
        Shape of one basis function value; ``(1,)`` for a scalar element.
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        Local DOF ids owned by each reference-cell entity, keyed by entity
        dimension (0 vertices, 1 facets, 2 cell interior), with one tuple per
        entity.
    is_nodal : bool
        Whether every DOF is point evaluation at its entity's coordinate.
    mapping : str
        How reference values are mapped to physical values: ``"identity"``
        or ``"contravariant_piola"``.
    is_orientation_dependent : bool
        Whether a DOF's sign depends on the orientation of its entity.

    Notes
    -----
    An element knows nothing about global numbering or assembly.
    ``FiniteElementSpace`` reads ``entity_dofs`` to number global DOFs,
    place DOF coordinates, and find boundary DOFs, without element-specific
    code. Every element here assigns at most one DOF per entity; for P1,
    ``entity_dofs`` is ``{0: ((0,), (1,), (2,)), 1: ((), (), ()), 2: ((),)}``.

    ``is_nodal`` is required by ``interpolate_function``; a non-nodal element
    such as RT0, whose DOFs are flux moments, needs its own interpolation.

    With ``mapping == "identity"`` physical values equal reference values and
    ``CellBasis.gradients`` holds ``grad_ref @ J^-1``. With
    ``"contravariant_piola"`` physical values are ``(1 / det J) J
    reference_value``, they depend on the cell, and ``CellBasis.divergences``
    holds the physical divergence.

    Orientation-dependent DOFs (facet fluxes) get a sign in
    ``FiniteElementSpace.local_dof_signs`` so that a shared facet's global
    DOF has one meaning regardless of which cell computes it.
    """

    reference_cell: ReferenceCell
    polynomial_degree: int
    number_of_local_dofs: int
    value_shape: tuple[int, ...]
    entity_dofs: dict[int, tuple[tuple[int, ...], ...]]
    is_nodal: bool
    mapping: str
    is_orientation_dependent: bool

    def tabulate_basis_values(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate all local basis functions at reference points.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, d)``.

        Returns
        -------
        basis_values : jax.Array
            Basis values, shape ``(B, Q, N_test, value_dim)``.

        Notes
        -----
        ``B`` is usually one (the same quadrature points for every cell), but
        any batch size must be accepted; ``FacetBasis`` passes one set of
        points per facet.
        """
        ...

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate the reference gradients of all local basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, d)``.

        Returns
        -------
        basis_gradients : jax.Array
            Reference gradients, shape ``(B, Q, N_test, d)``, or
            ``(B, 1, N_test, d)`` when constant in the point (e.g. P1).
        """
        ...
