"""
Lowest-order Raviart-Thomas element on the reference triangle.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem.reference_cell.triangle import ReferenceTriangle

_RT0_ENTITY_DOFS: dict[int, tuple[tuple[int, ...], ...]] = {
    0: ((), (), ()),
    1: ((0,), (1,), (2,)),
    2: ((),),
}

# Every reference RT0 basis function is (a, b) + (x, y), whose divergence is 2.
_REFERENCE_DIVERGENCE = 2.0


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class RaviartThomasTriangleRT0:
    r"""
    Lowest-order Raviart-Thomas H(div) element on the reference triangle.

    Attributes
    ----------
    reference_cell : ReferenceTriangle
        The reference triangle the element is defined on.
    polynomial_degree : int
        Always ``0``.
    number_of_local_dofs : int
        Always ``3``.
    value_shape : tuple[int, ...]
        Always ``(2,)``.
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        ``{0: ((), (), ()), 1: ((0,), (1,), (2,)), 2: ((),)}``; every DOF is
        owned by a facet.
    is_nodal : bool
        Always ``False``.
    mapping : str
        Always ``"contravariant_piola"``.
    is_orientation_dependent : bool
        Always ``True``.

    Notes
    -----
    The reference basis is

    .. math::

        \hat\psi_0 = (\hat x, \hat y), \quad
        \hat\psi_1 = (\hat x - 1, \hat y), \quad
        \hat\psi_2 = (\hat x, \hat y - 1),

    and local DOF ``i`` is the signed normal flux through facet ``i``. The
    element is not nodal, so ``interpolate_function`` does not apply; RT0
    fields come from solving the mixed system. Reversing an edge flips the
    sign of its flux, so ``FiniteElementSpace`` assigns each shared facet one
    global orientation (``local_dof_signs``).

    This class tabulates reference quantities only. ``create_cell_basis``
    applies the contravariant Piola map

    .. math::

        \psi(x) = \frac{1}{\det J_K} J_K \hat\psi(\hat x), \qquad
        \nabla \cdot \psi(x) = \frac{1}{\det J_K}
            \widehat\nabla \cdot \hat\psi(\hat x).
    """

    reference_cell: ReferenceTriangle
    polynomial_degree: int = dataclasses.field(default=0, init=False)
    number_of_local_dofs: int = dataclasses.field(default=3, init=False)
    value_shape: tuple[int, ...] = dataclasses.field(default=(2,), init=False)
    entity_dofs: dict[int, tuple[tuple[int, ...], ...]] = dataclasses.field(
        default_factory=lambda: _RT0_ENTITY_DOFS, init=False
    )
    is_nodal: bool = dataclasses.field(default=False, init=False)
    mapping: str = dataclasses.field(default="contravariant_piola", init=False)
    is_orientation_dependent: bool = dataclasses.field(default=True, init=False)

    def tabulate_basis_values(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate the reference basis functions at reference points.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_values : jax.Array
            Reference basis values, shape ``(B, Q, 3, 2)``.

        Notes
        -----
        Physical values need the contravariant Piola map, applied by
        ``create_cell_basis``.
        """
        _validate_reference_points(reference_points)
        x = reference_points[..., [0]]
        y = reference_points[..., [1]]
        psi_0 = jnp.concatenate((x, y), axis=-1)
        psi_1 = jnp.concatenate((x - 1.0, y), axis=-1)
        psi_2 = jnp.concatenate((x, y - 1.0), axis=-1)
        return jnp.concatenate((psi_0, psi_1, psi_2), axis=-2)

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """
        Reference gradients; not defined for this vector-valued element.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

        Raises
        ------
        NotImplementedError
            Always; use ``tabulate_basis_divergences`` instead.
        """
        raise NotImplementedError(
            "RaviartThomasTriangleRT0 has no tabulate_basis_gradients: it is "
            "vector-valued, so the relevant differential quantity is the "
            "divergence, not the gradient. Use tabulate_basis_divergences."
        )

    def tabulate_basis_divergences(self, reference_points: jax.Array) -> jax.Array:
        """
        Evaluate the reference divergences of the basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Reference points, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_divergences : jax.Array
            Reference divergences, all equal to ``2``, shape ``(B, 1, 3, 1)``.

        Notes
        -----
        The divergences are constant, so the quadrature axis has size one.
        Physical divergences are scaled by ``1 / det(J)`` in
        ``create_cell_basis``.
        """
        _validate_reference_points(reference_points)
        return jnp.full(
            (reference_points.shape[-4], 1, 3, 1), _REFERENCE_DIVERGENCE
        )


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
