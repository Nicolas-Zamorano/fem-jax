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

# Edge-midpoint DOF 3 + i is owned by facet i, the edge opposite local
# vertex i, matching ReferenceTriangle.facets_to_vertices.
_P2_ENTITY_DOFS: dict[int, tuple[tuple[int, ...], ...]] = {
    0: ((0,), (1,), (2,)),
    1: ((3,), (4,), (5,)),
    2: ((),),
}


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class LagrangeTriangleP1:
    """
    Scalar nodal P1 Lagrange element on the reference triangle.

    Attributes
    ----------
    reference_cell : ReferenceTriangle
        The reference triangle the element is defined on.
    polynomial_degree : int
        Always ``1``.
    number_of_local_dofs : int
        Always ``3``.
    value_shape : tuple[int, ...]
        Always ``(1,)``.
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        ``{0: ((0,), (1,), (2,)), 1: ((), (), ()), 2: ((),)}``; every DOF is
        owned by a vertex.
    is_nodal : bool
        Always ``True``.
    mapping : str
        Always ``"identity"``.
    is_orientation_dependent : bool
        Always ``False``.

    Notes
    -----
    The basis functions are the barycentric coordinates of the reference
    triangle, and local DOF ``i`` is point evaluation at reference vertex
    ``i``. With vertex-owned DOFs, ``cells_to_dofs`` equals
    ``cells_to_vertices``.
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
        """
        Evaluate the basis functions at reference points.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_values : jax.Array
            Basis values, shape ``(B, Q, 3, 1)``.

        Notes
        -----
        ``B`` is usually one (the same quadrature points for every cell), but
        any batch size is accepted; ``FacetBasis`` passes one set of points
        per facet.
        """
        _validate_reference_points(reference_points)
        x = reference_points[..., [0]]
        y = reference_points[..., [1]]
        return jnp.concatenate((1.0 - x - y, x, y), axis=-2)

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate the reference gradients of the basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_gradients : jax.Array
            Reference gradients, shape ``(B, 1, 3, 2)``.

        Notes
        -----
        The gradients are constant, so the quadrature axis has size one and
        broadcasts wherever it is used.
        """
        _validate_reference_points(reference_points)
        reference_gradients = jnp.asarray(_P1_REFERENCE_GRADIENTS)
        return jnp.broadcast_to(
            reference_gradients, (reference_points.shape[-4], 1, 3, 2)
        )


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class LagrangeTriangleP2:
    """
    Scalar nodal P2 Lagrange element on the reference triangle.

    Attributes
    ----------
    reference_cell : ReferenceTriangle
        The reference triangle the element is defined on.
    polynomial_degree : int
        Always ``2``.
    number_of_local_dofs : int
        Always ``6``.
    value_shape : tuple[int, ...]
        Always ``(1,)``.
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        ``{0: ((0,), (1,), (2,)), 1: ((3,), (4,), (5,)), 2: ((),)}``.
    is_nodal : bool
        Always ``True``.
    mapping : str
        Always ``"identity"``.
    is_orientation_dependent : bool
        Always ``False``.

    Notes
    -----
    With barycentric coordinates ``l0 = 1 - x - y``, ``l1 = x``, ``l2 = y``,
    the vertex functions are ``phi_i = li (2 li - 1)`` and the edge-midpoint
    functions are ``phi_3 = 4 l1 l2``, ``phi_4 = 4 l2 l0``, ``phi_5 = 4 l0 l1``,
    owned by facets 0, 1, 2. Midpoint DOFs are symmetric in the edge's two
    endpoints, so no orientation correction is needed. Reference gradients
    vary with the point, so the quadrature axis is not collapsed.
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
        """
        Evaluate the basis functions at reference points.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

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
        """
        Evaluate the reference gradients of the basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_gradients : jax.Array
            Reference gradients, shape ``(B, Q, 6, 2)``.
        """
        _validate_reference_points(reference_points)
        lambda_0, lambda_1, lambda_2 = _barycentric_coordinates(reference_points)

        def gradient(d_dx: jax.Array, d_dy: jax.Array) -> jax.Array:
            """
            Stack the two partial derivatives into a row gradient.

            Parameters
            ----------
            d_dx : jax.Array
                Derivative with respect to ``x``, shape ``(B, Q, 1, 1)``.
            d_dy : jax.Array
                Derivative with respect to ``y``, shape ``(B, Q, 1, 1)``.

            Returns
            -------
            gradient : jax.Array
                Row gradient, shape ``(B, Q, 1, 2)``.
            """
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
    """
    Barycentric coordinates of reference points.

    Parameters
    ----------
    reference_points : jax.Array
        Reference points, shape ``(B, Q, 1, 2)``.

    Returns
    -------
    barycentric_coordinates : tuple[jax.Array, jax.Array, jax.Array]
        ``(lambda_0, lambda_1, lambda_2)``, each of shape ``(B, Q, 1, 1)``.
    """
    x = reference_points[..., [0]]
    y = reference_points[..., [1]]
    return 1.0 - x - y, x, y


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
