"""
Residual-based a posteriori error estimator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.assembly.global_assembly import gather_vector
from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.function.evaluation import evaluate_finite_element_function
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.mesh.triangle import cell_diameters
from jax_fem.problem.elliptic import (
    EllipticCoefficientValues,
    EllipticProblem,
    TensorCoefficient,
    evaluate_elliptic_coefficients,
)
from jax_fem.space.cell_basis import CellBasis
from jax_fem.space.facet_basis import FacetBasis


def compute_element_residual_indicators(
    basis: CellBasis,
    coefficients: EllipticCoefficientValues,
    values: jax.Array,
    gradients: jax.Array,
    flux_divergence: jax.Array,
) -> jax.Array:
    """
    Squared element residual indicators.

    Parameters
    ----------
    basis : CellBasis
        Evaluated basis providing the cells and quadrature weights.
    coefficients : EllipticCoefficientValues
        Coefficients evaluated at ``basis.physical_points``; ``source``,
        ``advection``, and ``reaction`` are used.
    values : jax.Array
        Solution values ``u``, shape ``(K, Q, 1, 1)``.
    gradients : jax.Array
        Solution gradients ``grad u``, shape ``(K, Q, 1, d)`` or
        ``(K, 1, 1, d)``.
    flux_divergence : jax.Array
        Flux divergence ``div(A grad u)``, shape ``(K, Q, 1, 1)`` or
        ``(K, 1, 1, 1)``.

    Returns
    -------
    squared_indicators : jax.Array
        ``h_K^2 ||R_K||^2_{L^2(K)}`` for every cell, shape ``(K, 1, 1, 1)``.

    Notes
    -----
    The element residual is ``R_K = f + div(A grad u) - beta . grad u - c u``
    and ``h_K`` is the cell diameter. The solution quantities may come from a
    finite element function (see ``evaluate_flux_divergence``) or from any
    other representation evaluated at ``basis.physical_points``.
    """
    residual = (
        coefficients.source
        + flux_divergence
        - coefficients.advection @ gradients.mT
        - coefficients.reaction * values
    )
    squared_norms = integrate_cellwise(residual**2, basis.physical_weights)
    diameters = jnp.expand_dims(cell_diameters(basis.space.mesh), axis=(-3, -2, -1))
    return diameters**2 * squared_norms


def compute_facet_jump_indicators(
    facet_basis: FacetBasis,
    flux_plus: jax.Array,
    flux_minus: jax.Array,
) -> jax.Array:
    """
    Squared facet jump indicators.

    Parameters
    ----------
    facet_basis : FacetBasis
        Evaluated interior-facet geometry.
    flux_plus : jax.Array
        Flux ``A grad u`` from each facet's plus cell, shape ``(F, Q, 1, d)``.
    flux_minus : jax.Array
        Flux ``A grad u`` from each facet's minus cell, shape ``(F, Q, 1, d)``.

    Returns
    -------
    squared_indicators : jax.Array
        ``1/2 sum_e h_e ||J_e||^2_{L^2(e)}`` over the interior facets of
        every cell, shape ``(K, 1, 1, 1)``.

    Notes
    -----
    The normal flux jump is ``J_e = (flux_plus - flux_minus) . n_e``, with
    ``n_e`` pointing from the plus to the minus cell. Each facet contributes
    half of ``h_e ||J_e||^2`` to each of its two cells. Boundary facets
    contribute nothing.
    """
    normal_jumps = (flux_plus - flux_minus) @ facet_basis.normals.mT
    squared_norms = integrate_cellwise(normal_jumps**2, facet_basis.physical_weights)
    contributions = 0.5 * facet_basis.facet_lengths * squared_norms
    number_of_cells = facet_basis.space.mesh.cells_to_vertices.shape[-2]
    squared_indicators = jnp.zeros(
        contributions.shape[:-4] + (number_of_cells, 1, 1, 1),
        dtype=contributions.dtype,
    )
    squared_indicators = squared_indicators.at[
        ..., facet_basis.cells_plus, :, :, :
    ].add(contributions)
    return squared_indicators.at[..., facet_basis.cells_minus, :, :, :].add(
        contributions
    )


def evaluate_flux_divergence(
    solution: FiniteElementFunction,
    basis: CellBasis,
    diffusion: TensorCoefficient,
) -> jax.Array:
    """
    Divergence of the cellwise interpolant of the flux of a finite element function.

    Parameters
    ----------
    solution : FiniteElementFunction
        Finite element function on a nodal element, in ``basis.space``.
    basis : CellBasis
        Evaluated basis of ``solution.space``.
    diffusion : TensorCoefficient
        Diffusion tensor ``A``.

    Returns
    -------
    flux_divergence : jax.Array
        ``div I_K(A grad u_h)`` at the quadrature points, shape
        ``(K, Q, 1, 1)`` or ``(K, 1, 1, 1)``.

    Notes
    -----
    On each cell, the flux ``A grad u_h`` is evaluated at the cell's own
    nodal points (its local DOF coordinates) and interpolated in the
    vector-valued version of the same Lagrange element; the divergence of
    that interpolant needs only basis gradients. It equals
    ``div(A grad u_h)`` exactly whenever ``A grad u_h`` lies in the element
    space on each cell (e.g. constant or linear ``A`` for P1 and P2), and
    converges at ``O(h^k)`` in ``L^2`` otherwise. For P1 and constant ``A``
    it is exactly zero.
    """
    space = solution.space
    if space is not basis.space:
        raise ValueError(
            "solution and basis must share the same FiniteElementSpace instance."
        )
    if not space.element.is_nodal:
        raise NotImplementedError(
            "evaluate_flux_divergence requires a nodal element "
            "(element.is_nodal == True)."
        )

    mesh = space.mesh
    nodes = jnp.take(space.dof_coordinates, space.cells_to_dofs, axis=-3)
    origins = jnp.expand_dims(
        jnp.take(mesh.vertex_coordinates, mesh.cells_to_vertices[..., 0], axis=-3),
        axis=-3,
    )
    reference_nodes = (nodes - origins) @ basis.inverse_jacobians.mT
    node_gradients = (
        space.element.tabulate_basis_gradients(reference_nodes)
        @ basis.inverse_jacobians
    )
    gradients_at_nodes = gather_vector(solution.dof_values, space).mT @ node_gradients

    diffusion_at_nodes = diffusion(nodes)
    dimension = mesh.geometric_dimension
    check_shape(
        diffusion_at_nodes, (*nodes.shape[:-2], dimension, dimension), "diffusion(...)"
    )
    fluxes_at_nodes = gradients_at_nodes @ diffusion_at_nodes.mT
    nodal_fluxes = jnp.moveaxis(fluxes_at_nodes, -3, -2)
    return jnp.sum(nodal_fluxes * basis.gradients, axis=(-2, -1), keepdims=True)


def evaluate_facet_fluxes(
    solution: FiniteElementFunction,
    facet_basis: FacetBasis,
    diffusion: TensorCoefficient,
) -> tuple[jax.Array, jax.Array]:
    """
    Flux of a finite element function from both sides of every interior facet.

    Parameters
    ----------
    solution : FiniteElementFunction
        Finite element function in ``facet_basis.space``.
    facet_basis : FacetBasis
        Evaluated interior-facet geometry and gradients.
    diffusion : TensorCoefficient
        Diffusion tensor ``A``.

    Returns
    -------
    flux_plus : jax.Array
        ``A grad u_h`` from each facet's plus cell, shape ``(F, Q, 1, d)``.
    flux_minus : jax.Array
        ``A grad u_h`` from each facet's minus cell, shape ``(F, Q, 1, d)``.
    """
    if solution.space is not facet_basis.space:
        raise ValueError(
            "solution and facet_basis must share the same FiniteElementSpace "
            "instance."
        )

    local_dof_values = gather_vector(solution.dof_values, solution.space)
    diffusion_values = diffusion(facet_basis.physical_points)

    def side_flux(cells: jax.Array, gradients: jax.Array) -> jax.Array:
        """
        Flux from one side of every facet.

        Parameters
        ----------
        cells : jax.Array
            Cell on that side of each facet, shape ``(F,)``.
        gradients : jax.Array
            Basis gradients from that side, shape ``(F, Q, N_test, d)``.

        Returns
        -------
        flux : jax.Array
            ``A grad u_h`` from that side, shape ``(F, Q, 1, d)``.
        """
        side_dof_values = jnp.take(local_dof_values, cells, axis=-4)
        return (side_dof_values.mT @ gradients) @ diffusion_values.mT

    return (
        side_flux(facet_basis.cells_plus, facet_basis.gradients_plus),
        side_flux(facet_basis.cells_minus, facet_basis.gradients_minus),
    )


def compute_residual_error_estimator(
    solution: FiniteElementFunction,
    basis: CellBasis,
    facet_basis: FacetBasis,
    problem: EllipticProblem,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Residual-based error indicators and global estimator of a finite element solution.

    Parameters
    ----------
    solution : FiniteElementFunction
        Finite element solution on a nodal element.
    basis : CellBasis
        Evaluated basis of ``solution.space``.
    facet_basis : FacetBasis
        Evaluated interior-facet geometry of ``solution.space``.
    problem : EllipticProblem
        The elliptic problem.

    Returns
    -------
    cell_indicators : jax.Array
        Indicators ``eta_K``, shape ``(K, 1, 1, 1)``.
    global_estimator : jax.Array
        ``eta = sqrt(sum_K eta_K^2)``, shape ``()``.

    Notes
    -----
    .. math::

        \eta_K^2 = h_K^2 \|R_K\|^2_{L^2(K)}
            + \frac{1}{2} \sum_{e \in \partial K \setminus \partial\Omega}
              h_e \|J_e\|^2_{L^2(e)}

    combining ``compute_element_residual_indicators`` (with
    ``div(A grad u_h)`` from ``evaluate_flux_divergence``) and
    ``compute_facet_jump_indicators``. No Neumann boundary term is included.
    """
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    fields = evaluate_finite_element_function(solution, basis)
    flux_divergence = evaluate_flux_divergence(solution, basis, problem.diffusion)
    element_indicators = compute_element_residual_indicators(
        basis, coefficients, fields.values, fields.gradients, flux_divergence
    )
    flux_plus, flux_minus = evaluate_facet_fluxes(
        solution, facet_basis, problem.diffusion
    )
    jump_indicators = compute_facet_jump_indicators(facet_basis, flux_plus, flux_minus)
    squared_indicators = element_indicators + jump_indicators
    return jnp.sqrt(squared_indicators), jnp.sqrt(jnp.sum(squared_indicators))
