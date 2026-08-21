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
    Local approximation space, local basis, and local DOFs on a reference cell.

    A ``FiniteElement`` describes the reference-cell finite element only. It
    owns local basis functions and local degrees of freedom; it owns
    neither global numbering nor assembly. ``entity_dofs`` describes which
    reference-cell entities own its local DOFs, and is consumed generically
    by ``FiniteElementSpace`` for global DOF numbering, DOF coordinates, and
    boundary DOF extraction -- no per-element-family special casing.

    Attributes
    ----------
    reference_cell : ReferenceCell
        The reference cell this element is defined on.
    polynomial_degree: int
        Polynomial degree of the local approximation space.
    number_of_local_dofs: int
        Number of local degrees of freedom, ``N_phi``.
    value_shape : tuple[int, ...]
        Shape of one basis function's value; ``(1,)`` for a scalar element.
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        Local DOF ids owned by each reference-cell entity, keyed by entity
        dimension: ``0`` (the 3 vertices), ``1`` (the 3 facets/edges), ``2``
        (the cell interior, 1 entity). ``entity_dofs[dimension]`` has one
        tuple of local DOF ids per entity of that dimension, e.g. for
        ``LagrangeTriangleP1``, ``{0: ((0,), (1,), (2,)), 1: ((), (), ()),
        2: ((),)}``. Every element in this library assigns at most one DOF
        per entity; an element needing more (e.g. >= 2 DOFs per edge) is a
        future extension.
    is_nodal : bool
        Whether every local DOF is a point-evaluation functional at its
        entity's coordinate (true for Lagrange elements). Consumed by
        ``interpolate_function``, which requires it; a non-nodal element
        (e.g. ``RaviartThomasTriangleRT0``, whose DOFs are edge-flux
        moments) needs its own interpolation functionals instead.
    mapping : str
        How reference basis values are pushed forward to physical values,
        consumed by ``create_cell_basis``:

        - ``"identity"``: physical value == reference value (every scalar
          Lagrange element -- ``LagrangeTriangleP1``/``P2``/
          ``PiecewiseConstantTriangleP0``). ``CellBasis.gradients`` is the
          covariant-transformed (``grad_ref @ J^-1``) physical gradient;
          ``CellBasis.divergences`` is ``None``.
        - ``"contravariant_piola"``: physical value ==
          ``(1 / det(J)) J @ reference_value`` (``RaviartThomasTriangleRT0``,
          the standard H(div)-conforming Piola map). ``CellBasis.values``
          varies with the cell (unlike ``"identity"``, whose reference
          values are cell-independent); ``CellBasis.divergences`` is the
          physical divergence (``(1 / det(J)) *`` the reference divergence);
          ``CellBasis.gradients`` is ``None``.
    is_orientation_dependent : bool
        Whether a local DOF's sign depends on how its owning entity's local
        vertex ordering compares to the mesh's global canonical ordering
        (true only for facet-owned vector-flux DOFs, e.g. RT0's; false for
        every point-evaluation or cell-interior DOF, which have no
        direction to disagree on). When true, ``FiniteElementSpace`` computes
        a nontrivial ``local_dof_signs`` that ``create_cell_basis`` folds
        into the physical values/divergences, so a shared facet's global DOF
        has one unambiguous physical meaning regardless of which adjacent
        cell computed it.
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
            Points in reference coordinates, shape ``(B, Q, 1, d)`` for any
            batch size ``B``. ``B`` is normally ``1`` (the same reference
            points shared by every cell, e.g. a ``CellBasis``'s quadrature),
            but any ``B`` must be accepted: ``FacetBasis`` (Section 4.2)
            evaluates a distinct reference point per facet.

        Returns
        -------
        basis_values : jax.Array
            Basis values, shape ``(B, Q, N_phi, 1)``.
        """
        ...

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate the reference gradient of all local basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, d)`` (see
            ``tabulate_basis_values``).

        Returns
        -------
        basis_gradients : jax.Array
            Basis gradients with respect to reference coordinates, shape
            ``(B, Q, N_phi, d)`` or ``(B, 1, N_phi, d)`` if constant in
            ``Q`` (e.g. P1).
        """
        ...
