"""
Mixed (dual) forms for the H(div) x L^2 saddle-point system.
"""

import jax
import jax.numpy as jnp

from jax_fem.problem.elliptic import EllipticCoefficientValues
from jax_fem.space.cell_basis import CellBasis


def elliptic_mixed_mass_form(
    test_basis: CellBasis,
    trial_basis: CellBasis,
    coefficients: EllipticCoefficientValues,
) -> jax.Array:
    r"""
    Pointwise mixed mass form.

    Parameters
    ----------
    test_basis : CellBasis
        Evaluated H(div) test basis (``mapping == "contravariant_piola"``,
        e.g. RT0).
    trial_basis : CellBasis
        Evaluated H(div) trial basis, on the cells and quadrature points of
        ``test_basis``.
    coefficients : EllipticCoefficientValues
        Coefficients evaluated at ``test_basis.physical_points``; only
        ``diffusion`` is used.

    Returns
    -------
    pointwise_values : jax.Array
        Pointwise form values, shape ``(K, Q, N_test, N_trial)``.

    Notes
    -----
    Evaluates :math:`\tau \cdot A^{-1} \sigma` for every test function
    :math:`\tau` and trial function :math:`\sigma`.
    """
    diffusion_inverse = jnp.linalg.inv(coefficients.diffusion)
    return test_basis.values @ diffusion_inverse @ trial_basis.values.mT


def elliptic_mixed_div_form(
    test_basis: CellBasis,
    trial_basis: CellBasis,
    coefficients: None = None,
) -> jax.Array:
    r"""
    Pointwise mixed divergence form.

    Parameters
    ----------
    test_basis : CellBasis
        Evaluated scalar test basis (``mapping == "identity"``, e.g. P0).
    trial_basis : CellBasis
        Evaluated H(div) trial basis (``mapping == "contravariant_piola"``,
        e.g. RT0), on the cells and quadrature points of ``test_basis``.
    coefficients : None
        Unused; present to follow the ``BilinearForm`` protocol.

    Returns
    -------
    pointwise_values : jax.Array
        Pointwise form values, shape ``(K, Q, N_test, N_trial)``.

    Notes
    -----
    Evaluates :math:`v \, (\nabla \cdot \sigma)` for every test function
    :math:`v` and trial function :math:`\sigma`.
    """
    return test_basis.values @ trial_basis.divergences.mT
