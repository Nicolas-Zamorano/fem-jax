"""
Elliptic PDE problem data and coefficient evaluation.
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

#: Coefficient callables take physical points of shape ``(*batch, 1, d)`` and
#: return ``(*batch, 1, 1)`` (scalar), ``(*batch, 1, d)`` (vector), or
#: ``(*batch, d, d)`` (tensor). The three aliases are the same Python type and
#: only document which shape each ``EllipticProblem`` field returns.
ScalarCoefficient: TypeAlias = Callable[[jax.Array], jax.Array]
VectorCoefficient: TypeAlias = Callable[[jax.Array], jax.Array]
TensorCoefficient: TypeAlias = Callable[[jax.Array], jax.Array]


def zero_scalar_coefficient(points: jax.Array) -> jax.Array:
    """
    Scalar coefficient equal to zero everywhere.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, d)``.

    Returns
    -------
    values : jax.Array
        Zeros, shape ``(*batch, 1, 1)``.
    """
    return jnp.zeros(points.shape[:-1] + (1,), dtype=points.dtype)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class EllipticProblem:
    """
    Coefficient and boundary data of a second-order elliptic problem.

    Attributes
    ----------
    diffusion : TensorCoefficient
        Diffusion tensor ``A(x)``.
    advection : VectorCoefficient
        Advection field ``beta(x)``.
    reaction : ScalarCoefficient
        Reaction coefficient ``c(x)``.
    source : ScalarCoefficient
        Source term ``f(x)``.
    dirichlet_conditions : tuple[DirichletCondition, ...]
        Dirichlet boundary tags and their value functions.
    exact_solution : ScalarCoefficient | None
        Exact solution ``u``, for manufactured problems and error
        computation; not needed to solve.
    exact_gradient : VectorCoefficient | None
        Exact gradient ``grad u``, for the same purpose.
    advection_divergence : ScalarCoefficient
        Divergence ``div(beta)(x)`` of the advection field. Defaults to zero.
    name : str
        Name of the problem.

    Notes
    -----
    The problem is ``-div(A grad u) + beta . grad u + c u = f`` in the
    domain and ``u = g`` on the Dirichlet boundary. The fields store
    coefficient functions, not values evaluated on a particular basis.

    ``advection_divergence`` enters the symmetrized reaction
    ``c - div(beta) / 2`` used by the energy inner product and the energy
    error. The zero default is exact whenever ``div(beta) == 0`` (no
    advection, constant advection, or any divergence-free field). For an
    advection field with nonzero divergence it must be given; otherwise the
    energy form and the energy error are silently wrong.
    """

    diffusion: TensorCoefficient
    advection: VectorCoefficient
    reaction: ScalarCoefficient
    source: ScalarCoefficient
    dirichlet_conditions: tuple[DirichletCondition, ...]
    exact_solution: ScalarCoefficient | None = None
    exact_gradient: VectorCoefficient | None = None
    advection_divergence: ScalarCoefficient = zero_scalar_coefficient
    name: str = "elliptic_problem"


class EllipticCoefficientValues(NamedTuple):
    """
    Elliptic coefficients evaluated at a batch of physical points.

    Attributes
    ----------
    diffusion : jax.Array
        Diffusion tensor, shape ``(*batch, d, d)``.
    advection : jax.Array
        Advection field, shape ``(*batch, 1, d)``.
    reaction : jax.Array
        Reaction coefficient, shape ``(*batch, 1, 1)``.
    source : jax.Array
        Source term, shape ``(*batch, 1, 1)``.
    advection_divergence : jax.Array
        Divergence of the advection field, shape ``(*batch, 1, 1)``.
    """

    diffusion: jax.Array
    advection: jax.Array
    reaction: jax.Array
    source: jax.Array
    advection_divergence: jax.Array


def evaluate_elliptic_coefficients(
    problem: EllipticProblem, physical_points: jax.Array
) -> EllipticCoefficientValues:
    """
    Evaluate the elliptic coefficients at a batch of physical points.

    Parameters
    ----------
    problem : EllipticProblem
        The problem whose coefficient functions to evaluate.
    physical_points : jax.Array
        Physical points, shape ``(*batch, 1, d)``; ``*batch`` is ``(K, Q)``
        for cell quadrature points.

    Returns
    -------
    coefficients : EllipticCoefficientValues
        The evaluated coefficients, with shapes as documented on
        ``EllipticCoefficientValues``.
    """
    batch_shape = physical_points.shape[:-2]
    dimension = physical_points.shape[-1]

    diffusion = problem.diffusion(physical_points)
    advection = problem.advection(physical_points)
    reaction = problem.reaction(physical_points)
    source = problem.source(physical_points)
    advection_divergence = problem.advection_divergence(physical_points)

    check_shape(diffusion, (*batch_shape, dimension, dimension), "diffusion(...)")
    check_shape(advection, (*batch_shape, 1, dimension), "advection(...)")
    check_shape(reaction, (*batch_shape, 1, 1), "reaction(...)")
    check_shape(source, (*batch_shape, 1, 1), "source(...)")
    check_shape(
        advection_divergence, (*batch_shape, 1, 1), "advection_divergence(...)"
    )

    return EllipticCoefficientValues(
        diffusion=diffusion,
        advection=advection,
        reaction=reaction,
        source=source,
        advection_divergence=advection_divergence,
    )


def constant_scalar_coefficient(value: float) -> ScalarCoefficient:
    """
    Scalar coefficient with a constant value.

    Parameters
    ----------
    value : float
        The constant value.

    Returns
    -------
    coefficient : ScalarCoefficient
        Callable returning ``value`` with shape ``(*batch, 1, 1)``.
    """

    def coefficient(points: jax.Array) -> jax.Array:
        """
        Evaluate the constant at physical points.

        Parameters
        ----------
        points : jax.Array
            Physical points, shape ``(*batch, 1, d)``.

        Returns
        -------
        values : jax.Array
            Constant values, shape ``(*batch, 1, 1)``.
        """
        return jnp.full(points.shape[:-1] + (1,), value)

    return coefficient


def constant_vector_coefficient(vector: jax.Array) -> VectorCoefficient:
    """
    Vector coefficient with a constant value.

    Parameters
    ----------
    vector : jax.Array
        The constant vector, shape ``(d,)``.

    Returns
    -------
    coefficient : VectorCoefficient
        Callable returning ``vector`` with shape ``(*batch, 1, d)``.
    """

    def coefficient(points: jax.Array) -> jax.Array:
        """
        Evaluate the constant at physical points.

        Parameters
        ----------
        points : jax.Array
            Physical points, shape ``(*batch, 1, d)``.

        Returns
        -------
        values : jax.Array
            Constant vectors, shape ``(*batch, 1, d)``.
        """
        dimension = points.shape[-1]
        return jnp.broadcast_to(jnp.asarray(vector), points.shape[:-2] + (1, dimension))

    return coefficient


def constant_tensor_coefficient(tensor: jax.Array) -> TensorCoefficient:
    """
    Tensor coefficient with a constant value.

    Parameters
    ----------
    tensor : jax.Array
        The constant tensor, shape ``(d, d)``.

    Returns
    -------
    coefficient : TensorCoefficient
        Callable returning ``tensor`` with shape ``(*batch, d, d)``.
    """

    def coefficient(points: jax.Array) -> jax.Array:
        """
        Evaluate the constant at physical points.

        Parameters
        ----------
        points : jax.Array
            Physical points, shape ``(*batch, 1, d)``.

        Returns
        -------
        values : jax.Array
            Constant tensors, shape ``(*batch, d, d)``.
        """
        dimension = points.shape[-1]
        return jnp.broadcast_to(
            jnp.asarray(tensor), points.shape[:-2] + (dimension, dimension)
        )

    return coefficient


def isotropic_tensor_coefficient(scalar: ScalarCoefficient) -> TensorCoefficient:
    """
    Isotropic tensor coefficient ``a(x) I`` from a scalar coefficient.

    Parameters
    ----------
    scalar : ScalarCoefficient
        Scalar coefficient ``a(x)``.

    Returns
    -------
    coefficient : TensorCoefficient
        Callable returning ``a(x) I`` with shape ``(*batch, d, d)``.
    """

    def coefficient(points: jax.Array) -> jax.Array:
        """
        Evaluate ``a(x) I`` at physical points.

        Parameters
        ----------
        points : jax.Array
            Physical points, shape ``(*batch, 1, d)``.

        Returns
        -------
        values : jax.Array
            Isotropic tensors, shape ``(*batch, d, d)``.
        """
        dimension = points.shape[-1]
        return scalar(points) * jnp.eye(dimension)

    return coefficient
