"""
Fluxes exchanged between fractures through their traces.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.assembly.global_assembly import gather_vector
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.mesh.dfn import DfnMesh
from jax_fem.space.cell_basis import CellBasis


class TraceFluxes(NamedTuple):
    """
    Fracture outflows at points on every trace facet.

    Attributes
    ----------
    physical_points : jax.Array
        Points on each trace facet, in the order of ``mesh.trace_facets``,
        shape ``(N_trace_facets, S, 1, 3)``.
    arc_lengths : jax.Array
        Arc length of each point along its trace, shape
        ``(N_trace_facets, S, 1, 1)``.
    outflows : jax.Array
        Outflow of each of the trace's two fractures, in the order of
        ``mesh.traces_to_fractures``, shape ``(N_trace_facets, S, 1, 2)``.
    """

    physical_points: jax.Array
    arc_lengths: jax.Array
    outflows: jax.Array


def evaluate_trace_fluxes(
    solution: FiniteElementFunction,
    basis: CellBasis,
    transmissivity: jax.Array,
    facet_points: jax.Array,
) -> TraceFluxes:
    """
    Flux leaving each fracture through its traces.

    Parameters
    ----------
    solution : FiniteElementFunction
        Finite element function on a ``DfnMesh``, in ``basis.space``.
    basis : CellBasis
        Embedded basis of the space (see ``create_embedded_cell_basis``);
        only its inverse Jacobians are used.
    transmissivity : jax.Array
        Transmissivity tensor of each cell, shape ``(K, 1, 3, 3)``, e.g. from
        ``evaluate_fracture_transmissivity``.
    facet_points : jax.Array
        Points on the reference facet ``[0, 1]``, shape ``(1, S, 1, 1)``,
        such as the points of a ``ReferenceInterval`` quadrature rule.

    Returns
    -------
    trace_fluxes : TraceFluxes
        Points, arc lengths, and the outflows of both fractures.

    Raises
    ------
    TypeError
        If the mesh is not a ``DfnMesh``.
    ValueError
        If ``solution`` and ``basis`` have different spaces, or an array has
        the wrong shape.

    Notes
    -----
    The outflow of fracture ``i`` at a trace point is
    ``-sum_c (K_c grad u_h|_c) . nu_c`` over the cells ``c`` of fracture
    ``i`` on the facet, where ``nu_c`` is the unit conormal: in the plane of
    ``c``, perpendicular to the facet, and pointing out of ``c``. It is the
    Darcy flux ``-K grad u`` leaving fracture ``i`` through the trace,
    positive when fluid flows from fracture ``i`` into the other fracture.
    The flux balance at the trace makes the two outflows of the exact
    solution sum to zero; for ``u_h`` they do so up to the discretization
    error.

    Gradients are taken from each cell separately, so the outflows are
    constant on each facet for P1 and linear for P2 when the transmissivity
    is constant on each cell. Padded entries of ``mesh.facets_to_cells``
    are masked out.
    """
    space = solution.space
    mesh = space.mesh
    element = space.element

    if basis.space is not space:
        raise ValueError(
            "solution and basis must share the same FiniteElementSpace instance."
        )
    if not isinstance(mesh, DfnMesh):
        raise TypeError(
            f"evaluate_trace_fluxes needs a DfnMesh, got {type(mesh).__name__}."
        )
    number_of_cells = mesh.cells_to_vertices.shape[0]
    check_shape(transmissivity, (number_of_cells, 1, 3, 3), "transmissivity")
    check_shape(facet_points, (1, None, 1, 1), "facet_points")

    facets = mesh.trace_facets
    facet_vertices = mesh.facets_to_vertices[facets]
    first_vertices = jnp.expand_dims(
        mesh.vertex_coordinates[facet_vertices[..., 0]], axis=-3
    )
    second_vertices = jnp.expand_dims(
        mesh.vertex_coordinates[facet_vertices[..., 1]], axis=-3
    )
    edges = second_vertices - first_vertices
    tangents = edges / jnp.linalg.norm(edges, axis=-1, keepdims=True)
    midpoints = 0.5 * (first_vertices + second_vertices)
    physical_points = first_vertices + facet_points * edges

    traces = mesh.trace_facets_to_traces
    origins = jnp.expand_dims(mesh.trace_origins[traces], axis=-3)
    directions = jnp.expand_dims(mesh.trace_directions[traces], axis=-3)
    arc_lengths = (physical_points - origins) @ directions.mT

    local_dof_values = gather_vector(solution.dof_values, space)
    cell_origins = jnp.expand_dims(
        mesh.vertex_coordinates[mesh.cells_to_vertices[..., 0]], axis=-3
    )
    incident_cells = mesh.facets_to_cells[facets]
    vertex_sums = jnp.sum(facet_vertices, axis=-1)

    def cell_outflow(cells: jax.Array) -> jax.Array:
        """
        Outflow through each trace facet from one of its incident cells.

        Parameters
        ----------
        cells : jax.Array
            One incident cell of each trace facet, shape ``(N_trace_facets,)``.

        Returns
        -------
        outflow : jax.Array
            ``-(K grad u_h) . nu`` from that cell, shape
            ``(N_trace_facets, S, 1, 1)``.
        """
        inverse_jacobians = basis.inverse_jacobians[cells]
        reference_points = (
            physical_points - cell_origins[cells]
        ) @ inverse_jacobians.mT
        gradients = (
            element.tabulate_basis_gradients(reference_points) @ inverse_jacobians
        )
        solution_gradients = local_dof_values[cells].mT @ gradients
        fluxes = solution_gradients @ transmissivity[cells].mT

        opposite_vertices = (
            jnp.sum(mesh.cells_to_vertices[cells], axis=-1) - vertex_sums
        )
        outward = midpoints - jnp.expand_dims(
            mesh.vertex_coordinates[opposite_vertices], axis=-3
        )
        outward = outward - (outward @ tangents.mT) * tangents
        conormals = outward / jnp.linalg.norm(outward, axis=-1, keepdims=True)
        return -(fluxes @ conormals.mT)

    incidence_mask = mesh.facet_cell_mask[facets]
    cell_outflows = jnp.where(
        jnp.expand_dims(incidence_mask, axis=(-3, -2)),
        jnp.concatenate(
            [cell_outflow(incident_cells[..., slot]) for slot in range(4)], axis=-1
        ),
        0.0,
    )
    same_fracture = jnp.expand_dims(
        mesh.cells_to_fractures[incident_cells], axis=-1
    ) == jnp.expand_dims(mesh.traces_to_fractures[traces], axis=-2)
    selection = jnp.expand_dims(incidence_mask, axis=-1) & same_fracture
    outflows = cell_outflows @ jnp.expand_dims(
        selection.astype(cell_outflows.dtype), axis=-3
    )

    return TraceFluxes(
        physical_points=physical_points,
        arc_lengths=arc_lengths,
        outflows=jnp.broadcast_to(outflows, physical_points.shape[:-2] + (1, 2)),
    )
