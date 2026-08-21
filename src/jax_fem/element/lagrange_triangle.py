"""
Scalar P1 and P2 Lagrange elements on the reference triangle.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem.reference_cell.triangle import ReferenceTriangle

_P1_REFERENCE_GRADIENTS = ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))

_P1_ENTITY_DOFS: dict[int, tuple[tuple[int, ...], ...]] = {
    0: ((0,), (1,), (2,)),
    1: ((), (), ()),
    2: ((),),
}

# Local DOF 3 is owned by facet 0 (opposite local vertex 0, i.e. the v1-v2
# edge midpoint), local DOF 4 by facet 1 (v2-v0 edge midpoint), local DOF 5
# by facet 2 (v0-v1 edge midpoint) -- matching
# ReferenceTriangle.facets_to_vertices' facet numbering exactly.
_P2_ENTITY_DOFS: dict[int, tuple[tuple[int, ...], ...]] = {
    0: ((0,), (1,), (2,)),
    1: ((3,), (4,), (5,)),
    2: ((),),
}


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class LagrangeTriangleP1:
    """
    Scalar nodal P1 Lagrange element on the reference triangle.

    The three local basis functions are the barycentric coordinate
    functions of the reference triangle. Local DOF ``i`` is point
    evaluation at reference vertex ``i``, so local basis function ``i``
    satisfies ``phi_i(reference_vertex_j) = delta_ij``. Each local DOF is
    owned by the corresponding reference vertex (``entity_dofs``), and it is
    what lets ``FiniteElementSpace`` number ``cells_to_dofs`` identically to
    ``cells_to_vertices`` for this element.

    Attributes
    ----------
    reference_cell : ReferenceTriangle
        The ``ReferenceTriangle`` this element is defined on.
    polynomial_degree : int
        Always ``1``.
    number_of_local_dofs : int
        Always ``3``.
    value_shape : tuple[int, ...]
        Always ``(1,)`` (scalar-valued).
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        ``{0: ((0,), (1,), (2,)), 1: ((), (), ()), 2: ((),)}``: all 3 local
        DOFs are vertex-owned.
    is_nodal : bool
        Always ``True``.
    """

    reference_cell: ReferenceTriangle
    polynomial_degree: int = dataclasses.field(default=1, init=False)
    number_of_local_dofs: int = dataclasses.field(default=3, init=False)
    value_shape: tuple[int, ...] = dataclasses.field(default=(1,), init=False)
    entity_dofs: dict[int, tuple[tuple[int, ...], ...]] = dataclasses.field(
        default_factory=lambda: _P1_ENTITY_DOFS, init=False
    )
    is_nodal: bool = dataclasses.field(default=True, init=False)
    mapping: str = dataclasses.field(default="identity", init=False)
    is_orientation_dependent: bool = dataclasses.field(default=False, init=False)

    def tabulate_basis_values(self, reference_points: jax.Array) -> jax.Array:
        """Evaluate the three barycentric basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, 2)``. ``B``
            is normally ``1`` (the same reference points shared by every
            cell, e.g. a ``CellBasis``'s quadrature), but any ``B`` is
            accepted: ``FacetBasis`` evaluates a distinct set of reference
            points per facet (``B`` = number of facets), since each facet's
            physical quadrature points land at a different place within its
            own cell's reference triangle.

        Returns
        -------
        basis_values : jax.Array
            Basis values, shape ``(B, Q, 3, 1)``.
        """
        _validate_reference_points(reference_points)
        x = reference_points[..., [0]]
        y = reference_points[..., [1]]
        values = jnp.concatenate((1.0 - x - y, x, y), axis=-2)
        return values

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """Evaluate the (constant) reference gradients of the three basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, 2)`` (see
            ``tabulate_basis_values``).

        Returns
        -------
        basis_gradients : jax.Array
            Basis gradients, shape ``(B, 1, 3, 2)``. Constant for the affine
            P1 element, so it is not repeated along the quadrature axis;
            broadcasts against ``Q``-sized quantities wherever it is
            actually used (e.g. multiplied against the physical measure).
        """
        _validate_reference_points(reference_points)
        reference_gradients = jnp.asarray(_P1_REFERENCE_GRADIENTS)
        return jnp.broadcast_to(
            reference_gradients, (reference_points.shape[0], 1, 3, 2)
        )


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class LagrangeTriangleP2:
    """
    Scalar nodal P2 (quadratic) Lagrange element on the reference triangle.

    Six local basis functions in barycentric coordinates ``lambda_0 = 1 - x
    - y``, ``lambda_1 = x``, ``lambda_2 = y``:
    ``phi_i = lambda_i (2 lambda_i - 1)`` for the 3 vertex DOFs (``i = 0, 1,
    2``, point evaluation at reference vertex ``i``), and ``phi_3 = 4
    lambda_1 lambda_2``, ``phi_4 = 4 lambda_2 lambda_0``, ``phi_5 = 4
    lambda_0 lambda_1`` for the 3 edge-midpoint DOFs, owned by facets 0, 1,
    2 respectively (matching ``ReferenceTriangle.facets_to_vertices``'
    facet numbering: facet ``i`` opposite local vertex ``i``).

    An edge-midpoint coordinate and its quadratic shape function are both
    symmetric under endpoint reversal (``lambda_i lambda_j`` doesn't care
    which of the edge's two vertices is "first"), so these DOFs are
    orientation-invariant by construction: no edge-orientation permutation
    is needed, unlike an element with >= 2 DOFs per edge would require.

    Unlike P1, reference gradients vary with the quadrature point, so
    ``tabulate_basis_gradients`` does not collapse the quadrature axis.

    Attributes
    ----------
    reference_cell : ReferenceTriangle
        The ``ReferenceTriangle`` this element is defined on.
    polynomial_degree : int
        Always ``2``.
    number_of_local_dofs : int
        Always ``6``.
    value_shape : tuple[int, ...]
        Always ``(1,)`` (scalar-valued).
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        ``{0: ((0,), (1,), (2,)), 1: ((3,), (4,), (5,)), 2: ((),)}``.
    is_nodal : bool
        Always ``True``.
    """

    reference_cell: ReferenceTriangle
    polynomial_degree: int = dataclasses.field(default=2, init=False)
    number_of_local_dofs: int = dataclasses.field(default=6, init=False)
    value_shape: tuple[int, ...] = dataclasses.field(default=(1,), init=False)
    entity_dofs: dict[int, tuple[tuple[int, ...], ...]] = dataclasses.field(
        default_factory=lambda: _P2_ENTITY_DOFS, init=False
    )
    is_nodal: bool = dataclasses.field(default=True, init=False)
    mapping: str = dataclasses.field(default="identity", init=False)
    is_orientation_dependent: bool = dataclasses.field(default=False, init=False)

    def tabulate_basis_values(self, reference_points: jax.Array) -> jax.Array:
        """Evaluate the six quadratic basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, 2)`` for any
            batch size ``B`` (see ``LagrangeTriangleP1.tabulate_basis_values``).

        Returns
        -------
        basis_values : jax.Array
            Basis values, shape ``(B, Q, 6, 1)``.
        """
        _validate_reference_points(reference_points)
        lambda_0, lambda_1, lambda_2 = _barycentric_coordinates(reference_points)
        return jnp.concatenate(
            (
                lambda_0 * (2.0 * lambda_0 - 1.0),
                lambda_1 * (2.0 * lambda_1 - 1.0),
                lambda_2 * (2.0 * lambda_2 - 1.0),
                4.0 * lambda_1 * lambda_2,
                4.0 * lambda_2 * lambda_0,
                4.0 * lambda_0 * lambda_1,
            ),
            axis=-2,
        )

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """Evaluate the reference gradients of the six quadratic basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, 2)`` for any
            batch size ``B``.

        Returns
        -------
        basis_gradients : jax.Array
            Basis gradients, shape ``(B, Q, 6, 2)``. Unlike P1's constant
            gradients, these vary with the quadrature point, so the
            quadrature axis is not collapsed.
        """
        _validate_reference_points(reference_points)
        lambda_0, lambda_1, lambda_2 = _barycentric_coordinates(reference_points)

        def gradient(d_dx: jax.Array, d_dy: jax.Array) -> jax.Array:
            return jnp.concatenate((d_dx, d_dy), axis=-1)

        zero = jnp.zeros_like(lambda_0)
        gradients = (
            gradient(1.0 - 4.0 * lambda_0, 1.0 - 4.0 * lambda_0),
            gradient(4.0 * lambda_1 - 1.0, zero),
            gradient(zero, 4.0 * lambda_2 - 1.0),
            gradient(4.0 * lambda_2, 4.0 * lambda_1),
            gradient(-4.0 * lambda_2, 4.0 * (lambda_0 - lambda_2)),
            gradient(4.0 * (lambda_0 - lambda_1), -4.0 * lambda_1),
        )
        return jnp.concatenate(gradients, axis=-2)


def _barycentric_coordinates(
    reference_points: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(lambda_0, lambda_1, lambda_2)``, each shape ``(1, Q, 1, 1)``."""
    x = reference_points[..., [0]]
    y = reference_points[..., [1]]
    return 1.0 - x - y, x, y


def _validate_reference_points(reference_points: jax.Array) -> None:
    if reference_points.ndim != 4 or reference_points.shape[2:] != (1, 2):
        raise ValueError(
            "reference_points must have shape (B, Q, 1, 2) for some batch "
            f"size B, got {reference_points.shape}."
        )
