"""
Dirichlet condensation and solution reconstruction.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse

from jax_fem._shapes import check_shape
from jax_fem.assembly.global_assembly import GlobalMatrix
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.finite_element_space import FiniteElementSpace, find_boundary_dofs


def evaluate_dirichlet_dof_values(
    problem: EllipticProblem, space: FiniteElementSpace
) -> tuple[jax.Array, jax.Array]:
    """
    Indices and prescribed values of all Dirichlet DOFs.

    Parameters
    ----------
    problem : EllipticProblem
        The problem whose ``dirichlet_conditions`` are evaluated.
    space : FiniteElementSpace
        The finite element space the DOFs belong to.

    Returns
    -------
    dirichlet_dofs : jax.Array
        Dirichlet DOF indices in increasing order, shape ``(N_dirichlet,)``.
    dirichlet_values : jax.Array
        Prescribed values, shape ``(N_dirichlet, 1)``.

    Notes
    -----
    Each condition's value function is evaluated at the DOF coordinates of
    its boundary tags. A DOF reached by several conditions (e.g. a corner
    shared by two differently tagged edges) takes the value of the condition
    listed first in ``problem.dirichlet_conditions``.
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
        values = jnp.squeeze(raw_values, axis=-2)

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
    A linear system with Dirichlet DOFs eliminated, plus reconstruction data.

    Attributes
    ----------
    matrix : GlobalMatrix
        Free-free block ``A_ff``, shape ``(N_free, N_free)``.
    vector : jax.Array
        Right-hand side ``b_f - A_fc g_c``, shape ``(N_free, 1)``.
    free_dofs : jax.Array
        Global indices of the free DOFs, shape ``(N_free,)``.
    dirichlet_dofs : jax.Array
        Global indices of the Dirichlet DOFs, shape ``(N_dirichlet,)``.
    dirichlet_values : jax.Array
        Prescribed values at ``dirichlet_dofs``, shape ``(N_dirichlet, 1)``.
    number_of_dofs : int
        Total number of DOFs, ``N_free + N_dirichlet``.
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
    Eliminate Dirichlet DOFs from a global linear system.

    Parameters
    ----------
    matrix : GlobalMatrix
        Assembled global matrix, shape ``(dofs_test, dofs_test)``.
    vector : jax.Array
        Assembled global vector, shape ``(dofs_test, 1)``.
    dirichlet_dofs : jax.Array
        Global indices of the DOFs to eliminate, shape ``(N_dirichlet,)``.
    dirichlet_values : jax.Array
        Prescribed values at ``dirichlet_dofs``, shape ``(N_dirichlet, 1)``.

    Returns
    -------
    condensed_system : CondensedSystem
        The system ``A_ff u_f = b_f - A_fc g_c`` and the data needed to
        rebuild the full solution.

    Notes
    -----
    The matrix stays sparse: its entries are filtered by whether their row
    and column DOFs are free and reindexed into the free numbering.
    Duplicate ``(row, col)`` entries are preserved and summed on use, as in
    the unconstrained matrix.
    """
    number_of_dofs = matrix.shape[0]

    is_dirichlet = jnp.zeros((number_of_dofs,), dtype=bool)
    is_dirichlet = is_dirichlet.at[dirichlet_dofs].set(True)
    free_dofs = jnp.nonzero(~is_dirichlet)[0]

    free_index_of = jnp.full((number_of_dofs,), -1, dtype=jnp.int32)
    free_index_of = free_index_of.at[free_dofs].set(
        jnp.arange(free_dofs.shape[0], dtype=jnp.int32)
    )

    rows = matrix.indices[..., 0]
    columns = matrix.indices[..., 1]
    data = matrix.data

    row_is_free = ~is_dirichlet[rows]
    column_is_free = ~is_dirichlet[columns]

    free_free = row_is_free & column_is_free
    a_ff = jax_sparse.BCOO(
        (
            data[free_free],
            jnp.stack(
                (free_index_of[rows[free_free]], free_index_of[columns[free_free]]),
                axis=-1,
            ),
        ),
        shape=(free_dofs.shape[0], free_dofs.shape[0]),
    )

    # A_fc g_c: entries with a free row and a Dirichlet column, contracted
    # with the prescribed values and summed per free row.
    free_constrained = row_is_free & ~column_is_free
    dirichlet_value_of = jnp.zeros((number_of_dofs,))
    dirichlet_value_of = dirichlet_value_of.at[dirichlet_dofs].set(
        jnp.squeeze(dirichlet_values, axis=-1)
    )
    contributions = data[free_constrained] * dirichlet_value_of[
        columns[free_constrained]
    ]
    a_fc_g_c = jax.ops.segment_sum(
        contributions,
        free_index_of[rows[free_constrained]],
        num_segments=free_dofs.shape[0],
    )

    condensed_vector = vector[free_dofs] - jnp.expand_dims(a_fc_g_c, axis=-1)

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
    Rebuild the full DOF vector from the solved free DOFs.

    Parameters
    ----------
    free_dof_values : jax.Array
        Solution of the condensed system, shape ``(N_free, 1)``.
    condensed_system : CondensedSystem
        The condensed system that was solved.

    Returns
    -------
    dof_values : jax.Array
        Full DOF vector with the Dirichlet values inserted, shape
        ``(dofs_test, 1)``.
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
    return full.at[condensed_system.dirichlet_dofs].set(
        condensed_system.dirichlet_values
    )
