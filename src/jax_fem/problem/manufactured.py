"""
Manufactured elliptic problems for testing and convergence studies.

Each factory returns an ``EllipticProblem`` with a closed-form exact solution
and gradient, a source that makes it solve
``-div(A grad u) + beta . grad u + c u = f``, and Dirichlet data equal to the
exact solution on the whole boundary.
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


def _poisson_sin_sin_solution(points: jax.Array) -> jax.Array:
    """
    Exact solution ``sin(pi x) sin(pi y)``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Solution values, shape ``(*batch, 1, 1)``.
    """
    x, y = points[..., 0], points[..., 1]
    return jnp.expand_dims(jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), axis=-1)


def _poisson_sin_sin_gradient(points: jax.Array) -> jax.Array:
    """
    Gradient of ``sin(pi x) sin(pi y)``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    gradients : jax.Array
        Gradients, shape ``(*batch, 1, 2)``.
    """
    x, y = points[..., 0], points[..., 1]
    grad_x = jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
    grad_y = jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
    return jnp.stack((grad_x, grad_y), axis=-1)


def _poisson_sin_sin_source(points: jax.Array) -> jax.Array:
    """
    Source ``-Lap(u)`` for ``u = sin(pi x) sin(pi y)``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Source values, shape ``(*batch, 1, 1)``.
    """
    x, y = points[..., 0], points[..., 1]
    laplacian = -2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    return jnp.expand_dims(-laplacian, axis=-1)


def create_poisson_sin_sin_problem(mesh: TriangleMesh) -> EllipticProblem:
    """
    Poisson problem with ``u = sin(pi x) sin(pi y)`` on the unit square.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh of the unit square; its boundary tags carry the Dirichlet data.

    Returns
    -------
    problem : EllipticProblem
        ``-Lap(u) = f`` with ``A = I``, ``beta = 0``, ``c = 0``.

    Notes
    -----
    The solution vanishes on the whole boundary.
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


def _poisson_cos_sin_solution(points: jax.Array) -> jax.Array:
    """
    Exact solution ``cos(pi x) sin(pi y)``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Solution values, shape ``(*batch, 1, 1)``.
    """
    x, y = points[..., 0], points[..., 1]
    return jnp.expand_dims(jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y), axis=-1)


def _poisson_cos_sin_gradient(points: jax.Array) -> jax.Array:
    """
    Gradient of ``cos(pi x) sin(pi y)``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    gradients : jax.Array
        Gradients, shape ``(*batch, 1, 2)``.
    """
    x, y = points[..., 0], points[..., 1]
    grad_x = -jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    grad_y = jnp.pi * jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)
    return jnp.stack((grad_x, grad_y), axis=-1)


def _poisson_cos_sin_source(points: jax.Array) -> jax.Array:
    """
    Source ``-Lap(u)`` for ``u = cos(pi x) sin(pi y)``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Source values, shape ``(*batch, 1, 1)``.
    """
    x, y = points[..., 0], points[..., 1]
    laplacian = -2.0 * jnp.pi**2 * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
    return jnp.expand_dims(-laplacian, axis=-1)


def create_poisson_cos_sin_problem(mesh: TriangleMesh) -> EllipticProblem:
    """
    Poisson problem with ``u = cos(pi x) sin(pi y)`` on the unit square.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh of the unit square; its boundary tags carry the Dirichlet data.

    Returns
    -------
    problem : EllipticProblem
        ``-Lap(u) = f`` with ``A = I``, ``beta = 0``, ``c = 0`` and
        non-homogeneous Dirichlet data.
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


def _make_poisson_exponential_functions(scaling: float, exponential_constant: float):
    """
    Solution, gradient, and source of the boundary-layer problem.

    Parameters
    ----------
    scaling : float
        Amplitude of the solution.
    exponential_constant : float
        Rate ``k`` of the exponential boundary layer.

    Returns
    -------
    functions : tuple[ScalarCoefficient, VectorCoefficient, ScalarCoefficient]
        Solution, gradient, and source of
        ``u = scaling x y (1 - x)(1 - y)(exp(k x) - 1)``.
    """
    k = exponential_constant

    def solution(points: jax.Array) -> jax.Array:
        """
        Exact solution of the boundary-layer problem.

        Parameters
        ----------
        points : jax.Array
            Physical points, shape ``(*batch, 1, 2)``.

        Returns
        -------
        values : jax.Array
            Solution values, shape ``(*batch, 1, 1)``.
        """
        x, y = points[..., 0], points[..., 1]
        values = scaling * x * y * (1.0 - x) * (1.0 - y) * (jnp.exp(k * x) - 1.0)
        return jnp.expand_dims(values, axis=-1)

    def gradient(points: jax.Array) -> jax.Array:
        """
        Gradient of the boundary-layer solution.

        Parameters
        ----------
        points : jax.Array
            Physical points, shape ``(*batch, 1, 2)``.

        Returns
        -------
        gradients : jax.Array
            Gradients, shape ``(*batch, 1, 2)``.
        """
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
        """
        Source ``-Lap(u)`` of the boundary-layer problem.

        Parameters
        ----------
        points : jax.Array
            Physical points, shape ``(*batch, 1, 2)``.

        Returns
        -------
        values : jax.Array
            Source values, shape ``(*batch, 1, 1)``.
        """
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
        return jnp.expand_dims(-(uxx + uyy), axis=-1)

    return solution, gradient, source


def create_poisson_exponential_problem(
    mesh: TriangleMesh,
    *,
    scaling: float = 1.0,
    exponential_constant: float = 5.0,
) -> EllipticProblem:
    """
    Poisson problem with a boundary layer near ``x = 1``.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh of the unit square; its boundary tags carry the Dirichlet data.
    scaling : float
        Amplitude of the solution.
    exponential_constant : float
        Rate ``k`` of the boundary layer.

    Returns
    -------
    problem : EllipticProblem
        ``-Lap(u) = f`` with ``A = I``, ``beta = 0``, ``c = 0`` for
        ``u = scaling x y (1 - x)(1 - y)(exp(k x) - 1)``.

    Notes
    -----
    The solution vanishes on the whole boundary by construction.
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


_SINGULAR_ALPHA = 2.0 / 3.0
_SINGULAR_EPSILON = 1e-14


def _singular_polar(
    points: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Cartesian and polar quantities of the corner-singular solution.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    polar : tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(x, y, r, phi)`` with ``phi = 2/3 (theta + pi / 2)``, each of shape
        ``(*batch, 1)``.
    """
    x, y = points[..., 0], points[..., 1]
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    phi = _SINGULAR_ALPHA * (theta + 0.5 * jnp.pi)
    return x, y, r, phi


def _singular_gradient(
    x: jax.Array, y: jax.Array, r: jax.Array, phi: jax.Array
) -> jax.Array:
    """
    Gradient of ``r^(2/3) sin(phi)`` from its polar quantities.

    Parameters
    ----------
    x : jax.Array
        First coordinate, shape ``(*batch, 1)``.
    y : jax.Array
        Second coordinate, shape ``(*batch, 1)``.
    r : jax.Array
        Distance to the origin, shape ``(*batch, 1)``.
    phi : jax.Array
        Scaled angle ``2/3 theta'``, shape ``(*batch, 1)``.

    Returns
    -------
    gradients : jax.Array
        Gradients, shape ``(*batch, 1, 2)``, set to zero at the origin.

    Notes
    -----
    The true gradient is singular at the origin; the zero there is only a
    numerical placeholder.
    """
    safe_r = jnp.where(r > _SINGULAR_EPSILON, r, 1.0)
    cos_theta = x / safe_r
    sin_theta = y / safe_r
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    factor = _SINGULAR_ALPHA * safe_r ** (_SINGULAR_ALPHA - 1.0)

    grad_x = factor * (sin_phi * cos_theta - cos_phi * sin_theta)
    grad_y = factor * (sin_phi * sin_theta + cos_phi * cos_theta)
    gradient = jnp.stack((grad_x, grad_y), axis=-1)
    return jnp.where(jnp.expand_dims(r > _SINGULAR_EPSILON, axis=-1), gradient, 0.0)


def _poisson_singular_solution(points: jax.Array) -> jax.Array:
    """
    Exact solution ``r^(2/3) sin(2/3 (theta + pi / 2))``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Solution values, shape ``(*batch, 1, 1)``.
    """
    _, _, r, phi = _singular_polar(points)
    return jnp.expand_dims(r**_SINGULAR_ALPHA * jnp.sin(phi), axis=-1)


def _poisson_singular_gradient(points: jax.Array) -> jax.Array:
    """
    Gradient of ``r^(2/3) sin(2/3 (theta + pi / 2))``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    gradients : jax.Array
        Gradients, shape ``(*batch, 1, 2)``.
    """
    return _singular_gradient(*_singular_polar(points))


def create_poisson_singular_problem(mesh: TriangleMesh) -> EllipticProblem:
    """
    Harmonic problem with a singular gradient at the origin corner.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh of the unit square; its boundary tags carry the Dirichlet data.

    Returns
    -------
    problem : EllipticProblem
        ``-Lap(u) = 0`` with ``A = I``, ``beta = 0``, ``c = 0`` for
        ``u = r^(2/3) sin(2/3 (theta + pi / 2))``.

    Notes
    -----
    The origin is a convex corner of the unit square, so the solution stays
    finite there while its gradient does not. The Dirichlet data are the
    exact solution.
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


def _l_shaped_singular_polar(
    points: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Cartesian and polar quantities of the L-shaped re-entrant-corner solution.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    polar : tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(x, y, r, phi)`` with ``phi = 2/3 theta'``, each of shape
        ``(*batch, 1)``.

    Notes
    -----
    The domain ``[-1, 1]^2 \\ [0, 1] x [-1, 0]`` covers the angles from the
    positive x-axis counterclockwise to the negative y-axis. The angle is
    unwrapped to ``theta'`` in ``[0, 3 pi / 2]`` by adding ``2 pi`` to
    negative ``arctan2`` values; the removed fourth quadrant is never
    evaluated.
    """
    x, y = points[..., 0], points[..., 1]
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    theta_prime = jnp.where(theta >= 0.0, theta, theta + 2.0 * jnp.pi)
    phi = _SINGULAR_ALPHA * theta_prime
    return x, y, r, phi


def _poisson_l_shaped_singular_solution(points: jax.Array) -> jax.Array:
    """
    Exact solution ``r^(2/3) sin(2/3 theta')`` on the L-shaped domain.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Solution values, shape ``(*batch, 1, 1)``.
    """
    _, _, r, phi = _l_shaped_singular_polar(points)
    return jnp.expand_dims(r**_SINGULAR_ALPHA * jnp.sin(phi), axis=-1)


def _poisson_l_shaped_singular_gradient(points: jax.Array) -> jax.Array:
    """
    Gradient of ``r^(2/3) sin(2/3 theta')`` on the L-shaped domain.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    gradients : jax.Array
        Gradients, shape ``(*batch, 1, 2)``.
    """
    return _singular_gradient(*_l_shaped_singular_polar(points))


def create_poisson_l_shaped_singular_problem(mesh: TriangleMesh) -> EllipticProblem:
    """
    Harmonic problem with a re-entrant-corner singularity on the L-shaped domain.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh of ``[-1, 1]^2 \\ [0, 1] x [-1, 0]`` (see
        ``create_gmsh_l_shaped_mesh``).

    Returns
    -------
    problem : EllipticProblem
        ``-Lap(u) = 0`` with ``A = I``, ``beta = 0``, ``c = 0`` for
        ``u = r^(2/3) sin(2/3 theta')``.

    Notes
    -----
    ``theta'`` is measured counterclockwise from the positive x-axis and
    unwrapped to ``[0, 3 pi / 2]``. The solution vanishes on the two edges
    meeting at the re-entrant corner, and the Dirichlet data are the exact
    solution.
    """
    return EllipticProblem(
        name="poisson_l_shaped_singular",
        diffusion=constant_tensor_coefficient(_IDENTITY_2X2),
        advection=constant_vector_coefficient(_ZERO_VECTOR_2),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(
                mesh, _poisson_l_shaped_singular_solution
            ),
        ),
        exact_solution=_poisson_l_shaped_singular_solution,
        exact_gradient=_poisson_l_shaped_singular_gradient,
    )


def _simple_elliptic_solution(points: jax.Array) -> jax.Array:
    """
    Exact solution of the variable-coefficient problem.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Solution values, shape ``(*batch, 1, 1)``.
    """
    x, y = points[..., [0]], points[..., [1]]
    return jnp.sin(3.2 * x * (x - y)) * jnp.cos(x + 4.3 * y) + jnp.sin(
        4.6 * x + 9.2 * y
    ) * jnp.cos(5.2 * x - 2.6 * y)


def _simple_elliptic_gradient(points: jax.Array) -> jax.Array:
    """
    Gradient of the variable-coefficient solution.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    gradients : jax.Array
        Gradients, shape ``(*batch, 1, 2)``.
    """
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
    """
    Laplacian of the variable-coefficient solution.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Laplacian values, shape ``(*batch, 1)``.
    """
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
    """
    Scalar diffusion ``a = 2 + sin(x + 2y)``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Diffusion values, shape ``(*batch, 1, 1)``.
    """
    x, y = points[..., 0], points[..., 1]
    return jnp.expand_dims(2.0 + jnp.sin(x + 2.0 * y), axis=-1)


def _simple_elliptic_diffusion_gradient(points: jax.Array) -> jax.Array:
    """
    Gradient of the scalar diffusion ``a = 2 + sin(x + 2y)``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    gradients : jax.Array
        Gradients, shape ``(*batch, 1, 2)``.
    """
    x, y = points[..., 0], points[..., 1]
    d_dx = jnp.cos(x + 2.0 * y)
    d_dy = 2.0 * jnp.cos(x + 2.0 * y)
    return jnp.stack((d_dx, d_dy), axis=-1)


def _simple_elliptic_advection(points: jax.Array) -> jax.Array:
    """
    Advection ``beta = (sqrt(x - y^2 + 5), sqrt(y - x^2 + 5))``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Advection vectors, shape ``(*batch, 1, 2)``.
    """
    x, y = points[..., 0], points[..., 1]
    beta_x = jnp.sqrt(x - y**2 + 5.0)
    beta_y = jnp.sqrt(y - x**2 + 5.0)
    return jnp.stack((beta_x, beta_y), axis=-1)


def _simple_elliptic_advection_divergence(points: jax.Array) -> jax.Array:
    """
    Divergence of the advection field.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Divergence values, shape ``(*batch, 1, 1)``.
    """
    x, y = points[..., 0], points[..., 1]
    divergence = 0.5 / jnp.sqrt(x - y**2 + 5.0) + 0.5 / jnp.sqrt(y - x**2 + 5.0)
    return jnp.expand_dims(divergence, axis=-1)


def _simple_elliptic_reaction(points: jax.Array) -> jax.Array:
    """
    Reaction ``c = exp(x / 2 + y / 3) + 2``.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Reaction values, shape ``(*batch, 1, 1)``.
    """
    x, y = points[..., 0], points[..., 1]
    return jnp.expand_dims(jnp.exp(x / 2.0 + y / 3.0) + 2.0, axis=-1)


def _simple_elliptic_source(points: jax.Array) -> jax.Array:
    """
    Source of the variable-coefficient problem.

    Parameters
    ----------
    points : jax.Array
        Physical points, shape ``(*batch, 1, 2)``.

    Returns
    -------
    values : jax.Array
        Source values, shape ``(*batch, 1, 1)``.

    Notes
    -----
    For ``A = a I``, ``-div(A grad u) = -(a Lap(u) + grad(a) . grad(u))``.
    """
    laplacian = jnp.expand_dims(_simple_elliptic_laplacian(points), axis=-1)
    gradient = _simple_elliptic_gradient(points)
    diffusion_gradient = _simple_elliptic_diffusion_gradient(points)
    diffusion = _simple_elliptic_diffusion_scalar(points)
    advection = _simple_elliptic_advection(points)
    reaction = _simple_elliptic_reaction(points)
    solution = _simple_elliptic_solution(points)

    diffusion_term = diffusion * laplacian + jnp.sum(
        diffusion_gradient * gradient, axis=-1, keepdims=True
    )
    advection_term = jnp.sum(advection * gradient, axis=-1, keepdims=True)
    reaction_term = reaction * solution

    return -diffusion_term + advection_term + reaction_term


def create_simple_elliptic_problem(mesh: TriangleMesh) -> EllipticProblem:
    """
    Advection-diffusion-reaction problem with variable coefficients.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh of the unit square; its boundary tags carry the Dirichlet data.

    Returns
    -------
    problem : EllipticProblem
        Problem with isotropic diffusion ``A = a(x) I``, variable advection
        and reaction, and Dirichlet data equal to the exact solution.
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
