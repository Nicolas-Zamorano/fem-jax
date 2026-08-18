"""Manufactured elliptic problems for testing and convergence studies.

See Sections 12.1 and 18.3 of the architecture specification. Each factory
function below returns a complete ``EllipticProblem`` with a known closed-form
``exact_solution``/``exact_gradient`` and a hand-derived ``source`` that makes
it the exact solution of ``-div(A grad u) + beta . grad u + c u = f`` with
Dirichlet data equal to the exact solution on the whole boundary. These are
concrete, user-supplied manufactured problems, not a generic
autodiff-derived-source-term facility (Section 18.5 is a separate future
direction).

All five problems live on the unit square; build a compatible mesh with
``create_structured_unit_square_mesh`` (Section 7.3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_fem.mesh.triangle import TriangleMesh
from jax_fem.problem.boundary_conditions import create_full_boundary_dirichlet_condition
from jax_fem.problem.elliptic import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    isotropic_tensor_coefficient,
)

_IDENTITY_2X2 = jnp.eye(2)
_ZERO_VECTOR_2 = jnp.zeros(2)

# ---------------------------------------------------------------------------
# 1. u = sin(pi x) sin(pi y): homogeneous Dirichlet BC, constant coefficients.
# ---------------------------------------------------------------------------


def _poisson_sin_sin_solution(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    return (jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y))[..., None]


def _poisson_sin_sin_gradient(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    grad_x = jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
    grad_y = jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
    return jnp.stack((grad_x, grad_y), axis=-1)


def _poisson_sin_sin_source(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    laplacian = -2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    return (-laplacian)[..., None]


def create_poisson_sin_sin_problem(mesh: TriangleMesh) -> EllipticProblem:
    """``u = sin(pi x) sin(pi y)`` on the unit square, homogeneous Dirichlet BC.

    ``-Laplacian(u) = f``, with ``A = I``, ``beta = 0``, ``c = 0``. ``u``
    vanishes identically on the whole boundary.
    """
    return EllipticProblem(
        name="poisson_sin_sin",
        diffusion=constant_tensor_coefficient(_IDENTITY_2X2),
        advection=constant_vector_coefficient(_ZERO_VECTOR_2),
        reaction=constant_scalar_coefficient(0.0),
        source=_poisson_sin_sin_source,
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, _poisson_sin_sin_solution),
        ),
        exact_solution=_poisson_sin_sin_solution,
        exact_gradient=_poisson_sin_sin_gradient,
    )


# ---------------------------------------------------------------------------
# 2. u = cos(pi x) sin(pi y): non-homogeneous Dirichlet BC.
# ---------------------------------------------------------------------------


def _poisson_cos_sin_solution(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    return (jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y))[..., None]


def _poisson_cos_sin_gradient(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    grad_x = -jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    grad_y = jnp.pi * jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)
    return jnp.stack((grad_x, grad_y), axis=-1)


def _poisson_cos_sin_source(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    laplacian = -2.0 * jnp.pi**2 * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
    return (-laplacian)[..., None]


def create_poisson_cos_sin_problem(mesh: TriangleMesh) -> EllipticProblem:
    """``u = cos(pi x) sin(pi y)`` on the unit square, non-homogeneous Dirichlet BC.

    ``-Laplacian(u) = f``, with ``A = I``, ``beta = 0``, ``c = 0``.
    """
    return EllipticProblem(
        name="poisson_cos_sin",
        diffusion=constant_tensor_coefficient(_IDENTITY_2X2),
        advection=constant_vector_coefficient(_ZERO_VECTOR_2),
        reaction=constant_scalar_coefficient(0.0),
        source=_poisson_cos_sin_source,
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, _poisson_cos_sin_solution),
        ),
        exact_solution=_poisson_cos_sin_solution,
        exact_gradient=_poisson_cos_sin_gradient,
    )


# ---------------------------------------------------------------------------
# 3. Boundary-layer solution, homogeneous Dirichlet BC by construction.
# ---------------------------------------------------------------------------


def _make_poisson_exponential_functions(scaling: float, exponential_constant: float):
    k = exponential_constant

    def solution(points: jax.Array) -> jax.Array:
        x, y = points[..., 0], points[..., 1]
        values = scaling * x * y * (1.0 - x) * (1.0 - y) * (jnp.exp(k * x) - 1.0)
        return values[..., None]

    def gradient(points: jax.Array) -> jax.Array:
        x, y = points[..., 0], points[..., 1]
        exp_kx = jnp.exp(k * x)
        exm1 = exp_kx - 1.0
        one_x = 1.0 - x
        one_y = 1.0 - y

        grad_x = scaling * (
            y * one_y * (1.0 - 2.0 * x) * exm1 + x * y * one_x * one_y * k * exp_kx
        )
        grad_y = scaling * (x * one_x * (1.0 - 2.0 * y) * exm1)
        return jnp.stack((grad_x, grad_y), axis=-1)

    def source(points: jax.Array) -> jax.Array:
        x, y = points[..., 0], points[..., 1]
        exp_kx = jnp.exp(k * x)
        exm1 = exp_kx - 1.0
        one_x = 1.0 - x
        one_y = 1.0 - y

        uxx = (
            scaling
            * y
            * one_y
            * (
                -2.0 * exm1
                + 2.0 * (1.0 - 2.0 * x) * k * exp_kx
                + x * one_x * (k**2) * exp_kx
            )
        )
        uyy = -2.0 * scaling * x * one_x * exm1
        laplacian = uxx + uyy
        return (-laplacian)[..., None]

    return solution, gradient, source


def create_poisson_exponential_problem(
    mesh: TriangleMesh,
    *,
    scaling: float = 1.0,
    exponential_constant: float = 5.0,
) -> EllipticProblem:
    """Homogeneous-Dirichlet Poisson problem with a boundary layer near ``x = 1``.

    ``u = scaling * x y (1-x)(1-y)(exp(k x) - 1)``, ``k = exponential_constant``,
    which vanishes on the whole boundary by construction. ``A = I``,
    ``beta = 0``, ``c = 0``.
    """
    solution, gradient, source = _make_poisson_exponential_functions(
        scaling, exponential_constant
    )
    return EllipticProblem(
        name="poisson_exponential",
        diffusion=constant_tensor_coefficient(_IDENTITY_2X2),
        advection=constant_vector_coefficient(_ZERO_VECTOR_2),
        reaction=constant_scalar_coefficient(0.0),
        source=source,
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, solution),
        ),
        exact_solution=solution,
        exact_gradient=gradient,
    )


# ---------------------------------------------------------------------------
# 4. Singular harmonic solution (re-entrant-corner-style test problem).
# ---------------------------------------------------------------------------

_SINGULAR_ALPHA = 2.0 / 3.0
_SINGULAR_EPSILON = 1e-14


def _singular_polar(
    points: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x, y = points[..., 0], points[..., 1]
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    phi = _SINGULAR_ALPHA * (theta + 0.5 * jnp.pi)
    return x, y, r, phi


def _poisson_singular_solution(points: jax.Array) -> jax.Array:
    _, _, r, phi = _singular_polar(points)
    return (r**_SINGULAR_ALPHA * jnp.sin(phi))[..., None]


def _poisson_singular_gradient(points: jax.Array) -> jax.Array:
    x, y, r, phi = _singular_polar(points)
    # Avoid division by zero; the true gradient is singular at r = 0.
    safe_r = jnp.where(r > _SINGULAR_EPSILON, r, 1.0)

    cos_theta = x / safe_r
    sin_theta = y / safe_r
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    factor = _SINGULAR_ALPHA * safe_r ** (_SINGULAR_ALPHA - 1.0)

    grad_x = factor * (sin_phi * cos_theta - cos_phi * sin_theta)
    grad_y = factor * (sin_phi * sin_theta + cos_phi * cos_theta)
    gradient = jnp.stack((grad_x, grad_y), axis=-1)

    # This value at the origin is only a numerical placeholder: the true
    # gradient is not defined there.
    return jnp.where((r > _SINGULAR_EPSILON)[..., None], gradient, 0.0)


def create_poisson_singular_problem(mesh: TriangleMesh) -> EllipticProblem:
    """Harmonic solution with a singular gradient at the origin corner.

    ``u(r, theta) = r^(2/3) sin(2/3 (theta + pi/2))``, harmonic away from the
    origin (``source = 0``). ``A = I``, ``beta = 0``, ``c = 0``.
    Non-homogeneous Dirichlet BC equal to the exact solution. The origin is
    a corner of the unit square, so the solution stays finite there even
    though its gradient does not.
    """
    return EllipticProblem(
        name="poisson_singular",
        diffusion=constant_tensor_coefficient(_IDENTITY_2X2),
        advection=constant_vector_coefficient(_ZERO_VECTOR_2),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, _poisson_singular_solution),
        ),
        exact_solution=_poisson_singular_solution,
        exact_gradient=_poisson_singular_gradient,
    )


# ---------------------------------------------------------------------------
# 5. General advection-diffusion-reaction problem, all coefficients varying.
# ---------------------------------------------------------------------------


def _simple_elliptic_solution(points: jax.Array) -> jax.Array:
    x, y = points[..., [0]], points[..., [1]]
    values = jnp.sin(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y) + jnp.sin(
        4.6 * x + 9.2 * y
    ) * jnp.cos(5.2 * x - 2.6 * y)
    return values


def _simple_elliptic_gradient(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]

    t1_dx = (6.4 * x - 3.2 * y) * jnp.cos(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y)
    t2_dx = -jnp.sin(3.2 * x * (x - y)) * jnp.sin(x + 4.3 * y)
    t3_dx = -5.2 * jnp.sin(4.6 * x + 9.2 * y) * jnp.sin(5.2 * x - 2.6 * y)
    t4_dx = 4.6 * jnp.cos(4.6 * x + 9.2 * y) * jnp.cos(5.2 * x - 2.6 * y)

    t1_dy = -3.2 * x * jnp.cos(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y)
    t2_dy = -4.3 * jnp.sin(3.2 * x * (x - y)) * jnp.sin(x + 4.3 * y)
    t3_dy = 2.6 * jnp.sin(4.6 * x + 9.2 * y) * jnp.sin(5.2 * x - 2.6 * y)
    t4_dy = 9.2 * jnp.cos(4.6 * x + 9.2 * y) * jnp.cos(5.2 * x - 2.6 * y)

    grad_x = t1_dx + t2_dx + t3_dx + t4_dx
    grad_y = t1_dy + t2_dy + t3_dy + t4_dy
    return jnp.stack((grad_x, grad_y), axis=-1)


def _simple_elliptic_laplacian(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    uxx = (
        -((6.4 * x - 3.2 * y) ** 2) * jnp.sin(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y)
        - 2.0 * (6.4 * x - 3.2 * y) * jnp.sin(x + 4.3 * y) * jnp.cos(3.2 * x * (x - y))
        - jnp.sin(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y)
        - 48.2 * jnp.sin(4.6 * x + 9.2 * y) * jnp.cos(5.2 * x - 2.6 * y)
        - 47.84 * jnp.sin(5.2 * x - 2.6 * y) * jnp.cos(4.6 * x + 9.2 * y)
        + 6.4 * jnp.cos(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y)
    )
    uyy = (
        -10.24 * x**2 * jnp.sin(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y)
        + 27.52 * x * jnp.sin(x + 4.3 * y) * jnp.cos(3.2 * x * (x - y))
        - 18.49 * jnp.sin(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y)
        - 91.4 * jnp.sin(4.6 * x + 9.2 * y) * jnp.cos(5.2 * x - 2.6 * y)
        + 47.84 * jnp.sin(5.2 * x - 2.6 * y) * jnp.cos(4.6 * x + 9.2 * y)
    )
    return uxx + uyy


def _simple_elliptic_diffusion_scalar(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    return (2.0 + jnp.sin(x + 2.0 * y))[..., None]


def _simple_elliptic_diffusion_gradient(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    d_dx = jnp.cos(x + 2.0 * y)
    d_dy = 2.0 * jnp.cos(x + 2.0 * y)
    return jnp.stack((d_dx, d_dy), axis=-1)


def _simple_elliptic_advection(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    beta_x = jnp.sqrt(x - y**2 + 5.0)
    beta_y = jnp.sqrt(y - x**2 + 5.0)
    return jnp.stack((beta_x, beta_y), axis=-1)


def _simple_elliptic_advection_divergence(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    # d(beta_x)/dx + d(beta_y)/dy for beta = (sqrt(x-y^2+5), sqrt(y-x^2+5)).
    divergence = 0.5 / jnp.sqrt(x - y**2 + 5.0) + 0.5 / jnp.sqrt(y - x**2 + 5.0)
    return divergence[..., None]


def _simple_elliptic_reaction(points: jax.Array) -> jax.Array:
    x, y = points[..., 0], points[..., 1]
    return (jnp.exp(x / 2.0 + y / 3.0) + 2.0)[..., None]


def _simple_elliptic_source(points: jax.Array) -> jax.Array:
    laplacian = _simple_elliptic_laplacian(points)[..., None]  # (*b, 1, 1)
    gradient = _simple_elliptic_gradient(points)  # (*b, 1, 2)
    diffusion_gradient = _simple_elliptic_diffusion_gradient(points)  # (*b, 1, 2)
    diffusion = _simple_elliptic_diffusion_scalar(points)  # (*b, 1, 1)
    advection = _simple_elliptic_advection(points)  # (*b, 1, 2)
    reaction = _simple_elliptic_reaction(points)  # (*b, 1, 1)
    solution = _simple_elliptic_solution(points)  # (*b, 1, 1)

    # -div(A grad u) = -(a Laplacian(u) + grad(a) . grad(u)) for A = a * I.
    diffusion_term = diffusion * laplacian + jnp.sum(
        diffusion_gradient * gradient, axis=-1, keepdims=True
    )
    advection_term = jnp.sum(advection * gradient, axis=-1, keepdims=True)
    reaction_term = reaction * solution

    return -diffusion_term + advection_term + reaction_term


def create_simple_elliptic_problem(mesh: TriangleMesh) -> EllipticProblem:
    """General advection-diffusion-reaction problem with variable coefficients.

    Diffusion is isotropic (``A = a(x) I``) with a spatially varying scalar
    ``a``; advection and reaction also vary in space. Non-homogeneous
    Dirichlet BC equal to the exact solution.
    """
    return EllipticProblem(
        name="simple_elliptic_problem",
        diffusion=isotropic_tensor_coefficient(_simple_elliptic_diffusion_scalar),
        advection=_simple_elliptic_advection,
        reaction=_simple_elliptic_reaction,
        source=_simple_elliptic_source,
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, _simple_elliptic_solution),
        ),
        exact_solution=_simple_elliptic_solution,
        exact_gradient=_simple_elliptic_gradient,
        advection_divergence=_simple_elliptic_advection_divergence,
    )
