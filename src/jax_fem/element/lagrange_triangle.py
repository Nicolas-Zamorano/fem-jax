"""
Scalar P1 Lagrange element on the reference triangle.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem.reference_cell.triangle import ReferenceTriangle

_REFERENCE_GRADIENTS = ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class LagrangeTriangleP1:
    """
    Scalar nodal P1 Lagrange element on the reference triangle.

    The three local basis functions are the barycentric coordinate
    functions of the reference triangle. Local DOF ``i`` is point
    evaluation at reference vertex ``i``, so local basis function ``i``
    satisfies ``phi_i(reference_vertex_j) = delta_ij``. Each local DOF is
    owned by the corresponding reference vertex, and it is what lets
    ``FiniteElementSpace`` set ``cells_to_dofs = cells_to_vertices`` for
    this element.

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
    """

    reference_cell: ReferenceTriangle
    polynomial_degree: int = dataclasses.field(default=1, init=False)
    number_of_local_dofs: int = dataclasses.field(default=3, init=False)
    value_shape: tuple[int, ...] = dataclasses.field(default=(1,), init=False)

    def tabulate_basis_values(self, reference_points: jax.Array) -> jax.Array:
        """Evaluate the three barycentric basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(1, Q, 1, 2)``.

        Returns
        -------
        basis_values : jax.Array
            Basis values, shape ``(1, Q, 3, 1)``.
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
            Points in reference coordinates, shape ``(1, Q, 1, 2)``.

        Returns
        -------
        basis_gradients : jax.Array
            Basis gradients, shape ``(1, 1, 3, 2)``. Constant for the affine
            P1 element, so it is not repeated along the quadrature axis;
            broadcasts against ``Q``-sized quantities wherever it is
            actually used (e.g. multiplied against the physical measure).
        """
        _validate_reference_points(reference_points)
        reference_gradients = jnp.asarray(_REFERENCE_GRADIENTS)
        return jnp.broadcast_to(reference_gradients, (1, 1, 3, 2))


def _validate_reference_points(reference_points: jax.Array) -> None:
    if reference_points.ndim != 4 or reference_points.shape[0] != 1 or (
        reference_points.shape[2:] != (1, 2)
    ):
        raise ValueError(
            "reference_points must have shape (1, Q, 1, 2), got "
            f"{reference_points.shape}."
        )

