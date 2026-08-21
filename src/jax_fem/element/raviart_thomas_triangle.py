"""
Lowest-order Raviart-Thomas H(div)-conforming vector element on the reference
triangle.
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

# Every reference RT0 basis function has the form (a, b) + c*(x, y), with
# c == 1 for all three (Section 5.2); div((a,b) + c*(x,y)) = 2c == 2,
# constant, regardless of point or basis index.
_REFERENCE_DIVERGENCE = 2.0


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class RaviartThomasTriangleRT0:
    r"""
    Lowest-order Raviart-Thomas element on the reference triangle.

    Three vector-valued local basis functions, one per reference facet
    (edge), spanning ``P0(K_hat)^2 + x_hat P0(K_hat)`` (Section 5.2):

    .. math::

        \hat\psi_0 = (\hat x, \hat y), \quad
        \hat\psi_1 = (\hat x - 1, \hat y), \quad
        \hat\psi_2 = (\hat x, \hat y - 1)

    Local DOF ``i`` is the (signed) normal flux moment through reference
    facet ``i``, so each local DOF is owned by the corresponding facet
    (``entity_dofs``) -- unlike every Lagrange element in this library, RT0
    is **not** nodal (``is_nodal = False``: there is no point at which
    ``tabulate_basis_values`` reproduces the DOF functional, so
    ``interpolate_function`` cannot be used; RT0 fields are only ever
    obtained by solving the mixed system, Section 5.3) and **is**
    orientation-dependent (``is_orientation_dependent = True``: flipping an
    edge's traversal direction flips the sign of the flux it measures, so
    ``FiniteElementSpace`` must resolve a consistent global sign per shared
    facet, Section 5.2's ``S`` matrix).

    Physical basis functions use the contravariant Piola map
    (``mapping = "contravariant_piola"``, Section 5.2):

    .. math::

        \psi(x) = \frac{1}{\det J_K} J_K \hat\psi(\hat x), \qquad
        \nabla \cdot \psi(x) = \frac{1}{\det J_K}
            \widehat\nabla \cdot \hat\psi(\hat x)

    applied by ``create_cell_basis``, not by this class: this class only
    tabulates the reference-space quantities ``\hat\psi`` and
    ``\widehat\nabla \cdot \hat\psi``.

    Attributes
    ----------
    reference_cell : ReferenceTriangle
        The ``ReferenceTriangle`` this element is defined on.
    polynomial_degree : int
        Always ``0`` (lowest-order Raviart-Thomas).
    number_of_local_dofs : int
        Always ``3``.
    value_shape : tuple[int, ...]
        Always ``(2,)`` (vector-valued).
    entity_dofs : dict[int, tuple[tuple[int, ...], ...]]
        ``{0: ((), (), ()), 1: ((0,), (1,), (2,)), 2: ((),)}``: all 3 local
        DOFs are facet-owned.
    is_nodal : bool
        Always ``False``.
    mapping : str
        Always ``"contravariant_piola"``.
    is_orientation_dependent : bool
        Always ``True``.
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
        """Evaluate the three reference (un-mapped) RT0 vector fields.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_values : jax.Array
            Reference basis values, shape ``(B, Q, 3, 2)``. Physical values
            require the contravariant Piola map (``create_cell_basis``, not
            this method).
        """
        _validate_reference_points(reference_points)
        x = reference_points[..., [0]]
        y = reference_points[..., [1]]
        psi_0 = jnp.concatenate((x, y), axis=-1)
        psi_1 = jnp.concatenate((x - 1.0, y), axis=-1)
        psi_2 = jnp.concatenate((x, y - 1.0), axis=-1)
        return jnp.concatenate((psi_0, psi_1, psi_2), axis=-2)

    def tabulate_basis_gradients(self, reference_points: jax.Array) -> jax.Array:
        """Not defined for a vector-valued, Piola-mapped element.

        Raises
        ------
        NotImplementedError
            Always: RT0 is vector-valued, so "gradient" is not the relevant
            differential quantity -- use ``tabulate_basis_divergences``
            instead (``create_cell_basis`` does, for
            ``mapping == "contravariant_piola"`` elements).
        """
        raise NotImplementedError(
            "RaviartThomasTriangleRT0 has no tabulate_basis_gradients: it is "
            "vector-valued, so the relevant differential quantity is the "
            "divergence, not the gradient. Use tabulate_basis_divergences."
        )

    def tabulate_basis_divergences(self, reference_points: jax.Array) -> jax.Array:
        """Evaluate the (constant) reference divergence of the three basis functions.

        Parameters
        ----------
        reference_points : jax.Array
            Points in reference coordinates, shape ``(B, Q, 1, 2)``.

        Returns
        -------
        basis_divergences : jax.Array
            Reference divergences, shape ``(B, 1, 3, 1)``. Constant (every
            reference RT0 basis function has divergence ``2``), so the
            quadrature axis is not repeated; broadcasts against
            ``Q``-sized quantities wherever it is actually used. Physical
            divergences require an additional ``1 / det(J)`` scaling
            (``create_cell_basis``, not this method).
        """
        _validate_reference_points(reference_points)
        return jnp.full((reference_points.shape[0], 1, 3, 1), _REFERENCE_DIVERGENCE)


def _validate_reference_points(reference_points: jax.Array) -> None:
    if reference_points.ndim != 4 or reference_points.shape[2:] != (1, 2):
        raise ValueError(
            "reference_points must have shape (B, Q, 1, 2) for some batch "
            f"size B, got {reference_points.shape}."
        )
