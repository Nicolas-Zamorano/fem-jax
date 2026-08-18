"""Elliptic PDE problem data and coefficient evaluation.

See Sections 12.1 and 12.3 of the architecture specification.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape

if TYPE_CHECKING:
    from jax_fem.problem.boundary_conditions import DirichletCondition

#: A coefficient callable contract (Section 6.5): evaluated at physical
#: points of shape ``(*batch, 1, d)``, returning:
#:   scalar:  ``(*batch, 1, 1)``
#:   vector:  ``(*batch, 1, d)``
#:   tensor:  ``(*batch, d, d)``
#: The three aliases below are structurally identical Python types; they
#: exist to document which return shape each ``EllipticProblem`` field
#: expects.
ScalarCoefficient: TypeAlias = Callable[[jax.Array], jax.Array]
VectorCoefficient: TypeAlias = Callable[[jax.Array], jax.Array]
TensorCoefficient: TypeAlias = Callable[[jax.Array], jax.Array]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class EllipticProblem:
    """PDE coefficient and boundary data, independent of mesh, element, or quadrature.

    Stores coefficient **functions**, not values evaluated on a specific
    ``CellBasis`` (Section 12.1). The problem being discretized is the
    general second-order elliptic equation (Section 2):

    ``-div(A grad u) + beta . grad u + c u = f``  in the domain,
    ``u = g``  on the boundary.

    Attributes
    ----------
    diffusion:
        ``A(x)``, a ``TensorCoefficient``.
    advection:
        ``beta(x)``, a ``VectorCoefficient``.
    reaction:
        ``c(x)``, a ``ScalarCoefficient``.
    source:
        ``f(x)``, a ``ScalarCoefficient``.
    dirichlet_conditions:
        Which mesh boundary tags are Dirichlet and their value functions.
    exact_solution:
        Optional exact ``u`` for manufactured examples and error
        computation. Not required to solve the problem.
    exact_gradient:
        Optional exact ``grad u``, for the same purpose.
    advection_divergence:
        Optional ``div(beta)(x)``, a ``ScalarCoefficient``. Not required to
        solve the problem or to assemble any form: it exists solely for the
        energy-norm error diagnostic (``diagnostics.compute_energy_error``),
        whose coercive energy identity ``a(v, v) = ||v||_E^2`` needs the
        symmetrized reaction coefficient ``c - div(beta)/2``. Same optional,
        diagnostics-only status as ``exact_solution``/``exact_gradient``.
    """

    diffusion: TensorCoefficient
    advection: VectorCoefficient
    reaction: ScalarCoefficient
    source: ScalarCoefficient
    dirichlet_conditions: tuple[DirichletCondition, ...]
    exact_solution: ScalarCoefficient | None = None
    exact_gradient: VectorCoefficient | None = None
    advection_divergence: ScalarCoefficient | None = None


class EllipticCoefficientValues(NamedTuple):
    """Elliptic coefficients evaluated at a batch of physical points.

    Attributes
    ----------
    diffusion:
        Shape ``(*batch, d, d)``.
    advection:
        Shape ``(*batch, 1, d)``.
    reaction:
        Shape ``(*batch, 1, 1)``.
    source:
        Shape ``(*batch, 1, 1)``.
    """

    diffusion: jax.Array
    advection: jax.Array
    reaction: jax.Array
    source: jax.Array


def evaluate_elliptic_coefficients(
    problem: EllipticProblem, physical_points: jax.Array
) -> EllipticCoefficientValues:
    """Evaluate all four elliptic coefficients at a batch of physical points.

    Parameters
    ----------
    problem:
        The problem whose coefficient functions to evaluate.
    physical_points:
        Shape ``(*batch, 1, d)``. For cell quadrature evaluation,
        ``*batch = (K, Q)`` (Section 6.5).

    Returns
    -------
    EllipticCoefficientValues
        Shapes follow Section 6 exactly; see the class docstring.
    """

    batch_shape = physical_points.shape[:-2]
    dimension = physical_points.shape[-1]

    diffusion = problem.diffusion(physical_points)
    advection = problem.advection(physical_points)
    reaction = problem.reaction(physical_points)
    source = problem.source(physical_points)

    check_shape(diffusion, (*batch_shape, dimension, dimension), "diffusion(...)")
    check_shape(advection, (*batch_shape, 1, dimension), "advection(...)")
    check_shape(reaction, (*batch_shape, 1, 1), "reaction(...)")
    check_shape(source, (*batch_shape, 1, 1), "source(...)")

    return EllipticCoefficientValues(
        diffusion=diffusion, advection=advection, reaction=reaction, source=source
    )


def constant_scalar_coefficient(value: float) -> ScalarCoefficient:
    """Build a ``ScalarCoefficient`` constant in space, shape ``(*batch, 1, 1)``."""

    def coefficient(points: jax.Array) -> jax.Array:
        return jnp.full(points.shape[:-1] + (1,), value)

    return coefficient


def constant_vector_coefficient(vector: jax.Array) -> VectorCoefficient:
    """Build a ``VectorCoefficient`` constant in space, shape ``(*batch, 1, d)``.

    ``vector`` must have shape ``(d,)``.
    """

    def coefficient(points: jax.Array) -> jax.Array:
        dimension = points.shape[-1]
        return jnp.broadcast_to(
            jnp.asarray(vector), points.shape[:-2] + (1, dimension)
        )

    return coefficient


def constant_tensor_coefficient(tensor: jax.Array) -> TensorCoefficient:
    """Build a ``TensorCoefficient`` constant in space, shape ``(*batch, d, d)``.

    ``tensor`` must have shape ``(d, d)``.
    """

    def coefficient(points: jax.Array) -> jax.Array:
        dimension = points.shape[-1]
        return jnp.broadcast_to(
            jnp.asarray(tensor), points.shape[:-2] + (dimension, dimension)
        )

    return coefficient


def isotropic_tensor_coefficient(scalar: ScalarCoefficient) -> TensorCoefficient:
    """Lift a scalar coefficient ``a(x)`` to the isotropic tensor ``a(x) * I``.

    Used when a scalar (rather than fully anisotropic) diffusion coefficient
    is more natural to specify, e.g. for manufactured problems
    (``problem/manufactured.py``).
    """

    def coefficient(points: jax.Array) -> jax.Array:
        dimension = points.shape[-1]
        scalar_values = scalar(points)  # (*batch, 1, 1)
        return scalar_values * jnp.eye(dimension)

    return coefficient
