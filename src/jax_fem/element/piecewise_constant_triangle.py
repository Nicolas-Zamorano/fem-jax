"""
Scalar discontinuous P0 (piecewise-constant) element on the reference triangle.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem.reference_cell.triangle import ReferenceTriangle

_P0_ENTITY_DOFS: dict[int, tuple[tuple[int, ...], ...]] = {
    0: ((), (), ()),
    1: ((), (), ()),
    2: ((0,),),
}


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class PiecewiseConstantTriangleP0:
    """
    Scalar discontinuous P0 (piecewise-constant) element on the reference triangle.

    One local DOF per cell, ``entity_dofs`` cell-owned only (dimension 2):
    ``FiniteElementSpace`` therefore numbers one global DOF per mesh cell,
    with ``dof_coordinates`` at each cell's centroid (``entity_dofs``'
    generic dimension-2 handling, Section 3.2 -- unchanged for this
    element). Interpolation (``is_nodal = True``) evaluates a physical
    function at that centroid as an approximation to its true cell average;
    exact for any function already constant per cell (e.g. an interpolated
    ``PiecewiseConstantTriangleP0`` field itself), and to leading order in
    ``h`` otherwise -- consistent with the ``O(h)`` accuracy this element
    targets in the mixed formulation (Section 5).

    Attributes
    ----------
    reference_cell : ReferenceTriangle
        The ``ReferenceTriangle`` this element is defined on.
    polynomial_degree : int
        Always ``0``.
    number_of_local_dofs : int
        Always ``1``.
    value_shape : tuple[int, ...]
        Always ``(1,)`` (scalar-valued).
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        ``{0: ((), (), ()), 1: ((), (), ()), 2: ((0,),)}``: the single local
        DOF is cell-interior-owned.
    is_nodal : bool
        Always ``True`` (centroid evaluation, see above).
    mapping : str
        Always ``"identity"``.
    is_orientation_dependent : bool
        Always ``False``.
    """

    reference_cell: ReferenceTriangle
    polynomial_degree: int = dataclasses.field(default=0, init=False)
    number_of_local_dofs: int = dataclasses.field(default=1, init=False)
    value_shape: tuple[int, ...] = dataclasses.field(default=(1,), init=False)
    entity_dofs: dict[int, tuple[tuple[int, ...], ...]] = dataclasses.field(
        default_factory=lambda: _P0_ENTITY_DOFS, init=False
    )
    is_nodal: bool = dataclasses.field(default=True, init=False)
    mapping: str = dataclasses.field(default="identity", init=False)
    is_orientation_dependent: bool = dataclasses.field(default=False, init=False)

    def tabulate_basis_values(self, reference_points: jax.Array) -> jax.Array:
        """The single constant basis function, ``phi_0 = 1``.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_values : jax.Array
            Basis values, shape ``(B, Q, 1, 1)``, identically ``1``.
        """
        _validate_reference_points(reference_points)
        return jnp.ones(
            (reference_points.shape[0], reference_points.shape[1], 1, 1)
        )

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """The (zero) reference gradient of the constant basis function.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_gradients : jax.Array
            Basis gradients, shape ``(B, 1, 1, 2)``, identically ``0``.
        """
        _validate_reference_points(reference_points)
        return jnp.zeros((reference_points.shape[0], 1, 1, 2))


def _validate_reference_points(reference_points: jax.Array) -> None:
    if reference_points.ndim != 4 or reference_points.shape[2:] != (1, 2):
        raise ValueError(
            "reference_points must have shape (B, Q, 1, 2) for some batch "
            f"size B, got {reference_points.shape}."
        )
