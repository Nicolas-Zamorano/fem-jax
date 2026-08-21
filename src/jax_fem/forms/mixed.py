"""
Mixed (dual) formulation forms for the H(div) x L^2 saddle-point system.

See Section 5.3 of the implementation plan (``.context/implementation_plan.md``).
Unlike ``BilinearForm`` (Section 13), these forms do not conform to that
protocol: a mixed form's test and trial spaces genuinely differ (e.g. P0
test, RT0 trial), so its two ``CellBasis`` arguments and its
``(K, Q, N_phi_test, N_phi_trial)`` output need not be square.
"""

import jax
import jax.numpy as jnp

from jax_fem.problem.elliptic import EllipticCoefficientValues
from jax_fem.space.cell_basis import CellBasis


def elliptic_mixed_mass_form(
    basis_sigma: CellBasis, coefficients: EllipticCoefficientValues
) -> jax.Array:
    r"""
    Pointwise integrand of the mixed mass form.

    .. math:: a(\sigma, \tau) = \int_\Omega \sigma \cdot A^{-1} \tau \, dx

    Both test and trial functions are drawn from the same H(div)-conforming
    space (e.g. RT0), so -- unlike ``elliptic_mixed_div_form`` -- this *is*
    single-space and square, and can be assembled with the existing
    ``assemble_bilinear_form`` (Section 14.2) unchanged.

    Parameters
    ----------
    basis_sigma : CellBasis
        Evaluated H(div) geometry and basis (``mapping ==
        "contravariant_piola"``, e.g. RT0).
    coefficients : EllipticCoefficientValues
        Evaluated coefficients at ``basis_sigma.physical_points``; only
        ``.diffusion`` (``A``) is used.

    Returns
    -------
    local_matrix : jax.Array
        Shape ``(K, Q, N_phi, N_phi)``.
    """
    diffusion_inverse = jnp.linalg.inv(coefficients.diffusion)
    return basis_sigma.values @ diffusion_inverse @ basis_sigma.values.mT


def elliptic_mixed_div_form(basis_v: CellBasis, basis_sigma: CellBasis) -> jax.Array:
    r"""
    Pointwise integrand of the mixed divergence form.

    .. math:: b(v, \tau) = \int_\Omega v \, (\nabla \cdot \tau) \, dx

    Genuinely mixed: the test function ``v`` is drawn from the scalar space
    (e.g. P0) and the trial function's divergence from the H(div) space
    (e.g. RT0) -- two different spaces, so the result is not square in
    general and needs ``assemble_mixed_bilinear_form``
    (``assembly/mixed.py``), not ``assemble_bilinear_form``.

    Parameters
    ----------
    basis_v : CellBasis
        Evaluated scalar geometry and basis (``mapping == "identity"``,
        e.g. P0), the *test* space.
    basis_sigma : CellBasis
        Evaluated H(div) geometry and basis (``mapping ==
        "contravariant_piola"``, e.g. RT0), the *trial* space. Must share
        the same cells and quadrature points as ``basis_v`` (built from the
        same ``QuadratureRule``).

    Returns
    -------
    local_matrix : jax.Array
        Shape ``(K, Q, N_phi_v, N_phi_sigma)``.
    """
    return basis_v.values @ basis_sigma.divergences.mT
