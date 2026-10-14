"""
Scalar piecewise-constant P0 element on the reference triangle.
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
    Scalar piecewise-constant P0 element on the reference triangle.

    Attributes
    ----------
    reference_cell : ReferenceTriangle
        The reference triangle the element is defined on.
    polynomial_degree : int
        Always ``0``.
    number_of_local_dofs : int
        Always ``1``.
    value_shape : tuple[int, ...]
        Always ``(1,)``.
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        ``{0: ((), (), ()), 1: ((), (), ()), 2: ((0,),)}``; the single DOF is
        owned by the cell interior.
    is_nodal : bool
        Always ``True``.
    mapping : str
        Always ``"identity"``.
    is_orientation_dependent : bool
        Always ``False``.

    Notes
    -----
    The space has one DOF per cell, located at the cell centroid.
    Interpolation evaluates a function at the centroid, which approximates
    its cell average to leading order in ``h`` and is exact for functions
    that are already constant on each cell.
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
        """
        Evaluate the constant basis function at reference points.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_values : jax.Array
            Ones, shape ``(B, Q, 1, 1)``.
        """
        _validate_reference_points(reference_points)
        return jnp.ones(reference_points.shape[:-2] + (1, 1))

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate the reference gradient of the constant basis function.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_gradients : jax.Array
            Zeros, shape ``(B, 1, 1, 2)``.
        """
        _validate_reference_points(reference_points)
        return jnp.zeros((reference_points.shape[-4], 1, 1, 2))


def _validate_reference_points(reference_points: jax.Array) -> None:
    """
    Check that reference points have shape ``(B, Q, 1, 2)``.

    Parameters
    ----------
    reference_points : jax.Array
        Points to check.

    Raises
    ------
    ValueError
        If the shape is not ``(B, Q, 1, 2)``.
    """
    if reference_points.ndim != 4 or reference_points.shape[2:] != (1, 2):
        raise ValueError(
            "reference_points must have shape (B, Q, 1, 2) for some batch "
            f"size B, got {reference_points.shape}."
        )
