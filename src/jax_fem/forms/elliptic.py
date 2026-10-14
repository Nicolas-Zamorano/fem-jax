"""
The standard elliptic bilinear and linear forms.
"""

import jax

from jax_fem.problem.elliptic import EllipticCoefficientValues
from jax_fem.space.cell_basis import CellBasis


def elliptic_bilinear_form(
    test_basis: CellBasis,
    trial_basis: CellBasis,
    coefficients: EllipticCoefficientValues,
) -> jax.Array:
    """
    Pointwise elliptic bilinear form.

    Parameters
    ----------
    test_basis : CellBasis
        Evaluated test basis.
    trial_basis : CellBasis
        Evaluated trial basis, on the cells and quadrature points of
        ``test_basis``.
    coefficients : EllipticCoefficientValues
        Diffusion, advection, and reaction evaluated at
        ``test_basis.physical_points``.

    Returns
    -------
    pointwise_values : jax.Array
        Pointwise form values, shape ``(K, Q, N_test, N_trial)``; entry
        ``[..., i, j]`` couples test function ``i`` with trial function ``j``.

    Notes
    -----
    Evaluates ``(grad v)^T A grad u + v beta . grad u + c v u`` for every test
    function ``v`` and trial function ``u``. The test and trial bases may
    belong to different spaces (Petrov--Galerkin); pass the same basis twice
    for a Galerkin form.
    """
    diffusion = test_basis.gradients @ coefficients.diffusion @ trial_basis.gradients.mT
    trial_advection = trial_basis.gradients @ coefficients.advection.mT
    advection = test_basis.values @ trial_advection.mT
    reaction = coefficients.reaction * (test_basis.values @ trial_basis.values.mT)
    return diffusion + advection + reaction


def elliptic_linear_form(
    test_basis: CellBasis, coefficients: EllipticCoefficientValues
) -> jax.Array:
    """
    Pointwise elliptic linear form.

    Parameters
    ----------
    test_basis : CellBasis
        Evaluated test basis.
    coefficients : EllipticCoefficientValues
        Source evaluated at ``test_basis.physical_points``.

    Returns
    -------
    pointwise_values : jax.Array
        Pointwise form values, shape ``(K, Q, N_test, 1)``.

    Notes
    -----
    Evaluates ``f v`` for every test function ``v``.
    """
    return coefficients.source * test_basis.values


def elliptic_energy_bilinear_form(
    test_basis: CellBasis,
    trial_basis: CellBasis,
    coefficients: EllipticCoefficientValues,
) -> jax.Array:
    """
    Pointwise energy inner product of the elliptic operator.

    Parameters
    ----------
    test_basis : CellBasis
        Evaluated test basis.
    trial_basis : CellBasis
        Evaluated trial basis, on the cells and quadrature points of
        ``test_basis``.
    coefficients : EllipticCoefficientValues
        Diffusion, reaction, and advection divergence evaluated at
        ``test_basis.physical_points``.

    Returns
    -------
    pointwise_values : jax.Array
        Pointwise form values, shape ``(K, Q, N_test, N_trial)``.

    Notes
    -----
    Evaluates ``(grad v)^T A grad w + (c - div(beta) / 2) v w``, the
    symmetric part of the elliptic bilinear form on ``H^1_0``. Its Gram
    matrix is symmetric up to round-off, and positive definite on the free
    DOFs when ``A`` is symmetric positive definite, ``c - div(beta) / 2 >= 0``,
    and Dirichlet DOFs are removed.
    """
    diffusion = test_basis.gradients @ coefficients.diffusion @ trial_basis.gradients.mT
    symmetrized_reaction = (
        coefficients.reaction - 0.5 * coefficients.advection_divergence
    )
    reaction = symmetrized_reaction * (test_basis.values @ trial_basis.values.mT)
    return diffusion + reaction
