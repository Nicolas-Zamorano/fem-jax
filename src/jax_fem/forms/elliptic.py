"""
The standard elliptic bilinear and linear forms.
"""

import jax

from jax_fem.problem.elliptic import EllipticCoefficientValues
from jax_fem.space.cell_basis import CellBasis


def elliptic_bilinear_form(
    basis: CellBasis, coefficients: EllipticCoefficientValues
) -> jax.Array:
    """
    Pointwise integrand of the elliptic bilinear form.

    ``a(u, v) = int (grad v)^T A grad u + v beta . grad u + c v u``.

    Test function index ``i`` and trial function index ``j`` both range over
    local DOFs, giving the local matrix contribution at
    every cell and quadrature point.

    Parameters
    ----------
    basis: CellBasis
        Evaluated geometry and basis.
    coefficients: EllipticCoefficientValues
        Evaluated diffusion, advection, and reaction coefficients at
        ``basis.physical_points``.

    Returns
    -------
    local_matrix : jax.Array
        Shape ``(K, Q, N_phi, N_phi)``, entry ``[..., i, j]`` the
        contribution of test function ``i`` and trial function ``j``.
    """
    diffusion_integrand = (
        basis.gradients
        @ coefficients.diffusion
        @ basis.gradients.mT
    )

    advection_derivatives = basis.gradients @ coefficients.advection.mT
    advection_integrand = basis.values @ advection_derivatives.mT
    basis_value_products = basis.values @ basis.values.mT
    reaction_integrand = coefficients.reaction * basis_value_products

    return diffusion_integrand + advection_integrand + reaction_integrand


def elliptic_linear_form(
    basis: CellBasis, coefficients: EllipticCoefficientValues
) -> jax.Array:
    """Pointwise integrand of ``l(v) = int f v``.

    Parameters
    ----------
    basis: CellBasis
        Evaluated geometry and basis.
    coefficients: EllipticCoefficientValues
        Evaluated source coefficient at ``basis.physical_points``.

    Returns
    -------
    local_vector : jax.Array
        Shape ``(K, Q, N_phi, 1)``.
    """
    return coefficients.source * basis.values
