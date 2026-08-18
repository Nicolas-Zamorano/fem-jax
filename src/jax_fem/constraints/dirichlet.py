"""
Dirichlet condensation and solution reconstruction.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse

from jax_fem._shapes import check_shape
from jax_fem.assembly.matrix import GlobalMatrix
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.finite_element_space import FiniteElementSpace, find_boundary_dofs


def evaluate_dirichlet_dof_values(
    problem: EllipticProblem, space: FiniteElementSpace
) -> tuple[jax.Array, jax.Array]:
    """
    Evaluate every Dirichlet DOF's index and prescribed value.

    Uses mesh boundary tags (via ``find_boundary_dofs``), the space's DOF
    placement, and each ``DirichletCondition``'s value function. If a DOF is
    reachable from more than one condition (e.g. a mesh corner shared by two
    differently tagged edges), the condition listed first in
    ``problem.dirichlet_conditions`` wins.

    Parameters
    ----------
    problem : EllipticProblem
        The problem whose ``dirichlet_conditions`` to evaluate.
    space : FiniteElementSpace
        The finite element space the DOFs live in.

    Returns
    -------
    dof_values : tuple[jax.Array, jax.Array]
        Dirichlet DOF indices, shape ``(N_dirichlet,)``, in increasing
        order; and Dirichlet DOF values, shape ``(N_dirichlet, 1)``.
    """
    assigned = jnp.zeros((space.number_of_dofs,), dtype=bool)
    values_array = jnp.zeros((space.number_of_dofs, 1))

    for condition in problem.dirichlet_conditions:
        dofs = find_boundary_dofs(space, condition.boundary_tags)
        coordinates = space.dof_coordinates[dofs]
        raw_values = condition.value(coordinates)
        check_shape(
            raw_values,
            (dofs.shape[0], 1, 1),
            f"DirichletCondition(boundary_tags={condition.boundary_tags}).value(...)",
        )
        values = raw_values.squeeze(-2)

        not_yet_assigned = ~assigned[dofs]
        newly_assigned_dofs = dofs[not_yet_assigned]
        newly_assigned_values = values[not_yet_assigned]

        assigned = assigned.at[newly_assigned_dofs].set(True)
        values_array = values_array.at[newly_assigned_dofs].set(newly_assigned_values)

    dirichlet_dofs = jnp.nonzero(assigned)[0]
    dirichlet_values = values_array[dirichlet_dofs]
    return dirichlet_dofs, dirichlet_values


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class CondensedSystem:
    """
    A Dirichlet-eliminated algebraic system, plus reconstruction data.

    ``matrix`` and ``vector`` represent ``A_ff u_f = b_f - A_fc g_c``.

    Attributes
    ----------
    matrix : GlobalMatrix
        ``A_ff``, shape ``(N_free, N_free)``.
    vector : jax.Array
        ``b_f - A_fc g_c``, shape ``(N_free, 1)``.
    free_dofs : jax.Array
        Global DOF indices of the free unknowns, shape ``(N_free,)``.
    dirichlet_dofs : jax.Array
        Global DOF indices of the constrained DOFs, shape ``(N_dirichlet,)``.
    dirichlet_values: jax.Array
        Prescribed values at ``dirichlet_dofs``, shape ``(N_dirichlet, 1)``.
    number_of_dofs: int
        ``N_dofs = N_free + N_dirichlet``, the complete DOF count.
    """

    matrix: GlobalMatrix
    vector: jax.Array
    free_dofs: jax.Array
    dirichlet_dofs: jax.Array
    dirichlet_values: jax.Array
    number_of_dofs: int


def condense_dirichlet_system(
    matrix: GlobalMatrix,
    vector: jax.Array,
    dirichlet_dofs: jax.Array,
    dirichlet_values: jax.Array,
) -> CondensedSystem:
    """
    Eliminate constrained DOFs from a global linear system.

    Stays sparse throughout: entries of ``matrix`` are filtered by whether
    their row/column DOF is free or Dirichlet and reindexed into the free
    numbering directly, rather than densifying (Section 15.2). ``matrix``
    may hold duplicate ``(row, col)`` index pairs (Section 14.2); this
    filter-then-reindex approach preserves that, and duplicates are summed
    on materialization exactly as in the unconstrained matrix.

    Parameters
    ----------
    matrix:
        The assembled global bilinear-form matrix, shape ``(N_dofs, N_dofs)``.
    vector:
        The assembled global linear-form vector, shape ``(N_dofs, 1)``.
    dirichlet_dofs:
        Global DOF indices to eliminate, shape ``(N_dirichlet,)``.
    dirichlet_values:
        Prescribed values at ``dirichlet_dofs``, shape ``(N_dirichlet, 1)``.

    Returns
    -------
    CondensedSystem
    """
    number_of_dofs = matrix.shape[0]

    is_dirichlet = jnp.zeros((number_of_dofs,), dtype=bool)
    is_dirichlet = is_dirichlet.at[dirichlet_dofs].set(True)
    free_dofs = jnp.nonzero(~is_dirichlet)[0]

    free_index_of = jnp.full((number_of_dofs,), -1, dtype=jnp.int32)
    free_index_of = free_index_of.at[free_dofs].set(
        jnp.arange(free_dofs.shape[0], dtype=jnp.int32)
    )

    rows = matrix.indices[:, 0]
    cols = matrix.indices[:, 1]
    data = matrix.data

    row_is_free = ~is_dirichlet[rows]
    col_is_free = ~is_dirichlet[cols]

    # A_ff: both endpoints free, reindexed into the free numbering.
    ff_mask = row_is_free & col_is_free
    a_ff = jax_sparse.BCOO(
        (
            data[ff_mask],
            jnp.stack(
                (free_index_of[rows[ff_mask]], free_index_of[cols[ff_mask]]), axis=-1
            ),
        ),
        shape=(free_dofs.shape[0], free_dofs.shape[0]),
    )

    # A_fc @ g_c: free row, Dirichlet column, contracted against the known
    # values via a segment-sum over the (reindexed) free rows.
    fc_mask = row_is_free & ~col_is_free
    dirichlet_value_of = jnp.zeros((number_of_dofs,))
    dirichlet_value_of = dirichlet_value_of.at[dirichlet_dofs].set(
        dirichlet_values[:, 0]
    )
    fc_contribution = data[fc_mask] * dirichlet_value_of[cols[fc_mask]]
    a_fc_g_c = jax.ops.segment_sum(
        fc_contribution, free_index_of[rows[fc_mask]], num_segments=free_dofs.shape[0]
    )

    condensed_vector = (vector[free_dofs, 0] - a_fc_g_c)[:, None]  # (N_free, 1)

    return CondensedSystem(
        matrix=a_ff,
        vector=condensed_vector,
        free_dofs=free_dofs,
        dirichlet_dofs=dirichlet_dofs,
        dirichlet_values=dirichlet_values,
        number_of_dofs=number_of_dofs,
    )


def expand_condensed_solution(
    free_dof_values: jax.Array, condensed_system: CondensedSystem
) -> jax.Array:
    """
    Reconstruct the complete DOF vector from a solved free-DOF vector.

    Parameters
    ----------
    free_dof_values : jax.Array
        The user-solved free unknowns, shape ``(N_free, 1)``.
    condensed_system : CondensedSystem
        The ``CondensedSystem`` that produced the linear system solved for
        ``free_dof_values``.

    Returns
    -------
    jax.Array
        Shape ``(N_dofs, 1)``.
    """
    check_shape(
        free_dof_values,
        (condensed_system.free_dofs.shape[0], 1),
        "free_dof_values",
    )
    full = jnp.zeros(
        (condensed_system.number_of_dofs, 1), dtype=free_dof_values.dtype
    )
    full = full.at[condensed_system.free_dofs].set(free_dof_values)
    full = full.at[condensed_system.dirichlet_dofs].set(
        condensed_system.dirichlet_values
    )
    return full
