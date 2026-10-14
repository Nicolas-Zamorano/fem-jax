"""
Global assembly of local contributions, and its adjoint gather.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse

from jax_fem.space.finite_element_space import FiniteElementSpace

#: The library's sparse global matrix representation.
GlobalMatrix: TypeAlias = jax_sparse.BCOO


def assemble_matrix(
    local_matrices: jax.Array,
    test_space: FiniteElementSpace,
    trial_space: FiniteElementSpace,
) -> GlobalMatrix:
    """
    Assemble local matrices into a global sparse matrix.

    Parameters
    ----------
    local_matrices : jax.Array
        Integrated local matrices, shape ``(K, 1, N_test, N_trial)``.
    test_space : FiniteElementSpace
        Space indexing the matrix rows.
    trial_space : FiniteElementSpace
        Space indexing the matrix columns.

    Returns
    -------
    global_matrix : GlobalMatrix
        Assembled sparse matrix, shape ``(dofs_test, dofs_trial)``.

    Notes
    -----
    Contributions from cells sharing a DOF are stored as separate entries at
    the same ``(row, col)`` index and are not compacted. ``BCOO`` sums them in
    ``todense()``, matrix products, and other consuming operations; call
    ``.sum_duplicates()`` for a compacted representation. Batched global
    matrices are not supported; use ``jax.vmap`` over this function instead.
    """
    number_of_cells = test_space.cells_to_dofs.shape[-2]
    expected_shape = (
        number_of_cells,
        1,
        test_space.element.number_of_local_dofs,
        trial_space.element.number_of_local_dofs,
    )
    if local_matrices.shape != expected_shape:
        raise ValueError(
            f"local_matrices must have shape {expected_shape}, got "
            f"{local_matrices.shape}."
        )

    rows = jnp.broadcast_to(
        jnp.expand_dims(test_space.cells_to_dofs, axis=(-3, -1)),
        local_matrices.shape,
    )
    columns = jnp.broadcast_to(
        jnp.expand_dims(trial_space.cells_to_dofs, axis=(-3, -2)),
        local_matrices.shape,
    )
    indices = jnp.stack((jnp.ravel(rows), jnp.ravel(columns)), axis=-1)
    return jax_sparse.BCOO(
        (jnp.ravel(local_matrices), indices),
        shape=(test_space.number_of_dofs, trial_space.number_of_dofs),
    )


def assemble_vector(
    local_vectors: jax.Array, test_space: FiniteElementSpace
) -> jax.Array:
    """
    Assemble local vectors into a global vector.

    Parameters
    ----------
    local_vectors : jax.Array
        Integrated local vectors, shape ``(..., K, 1, N_test, 1)``.
    test_space : FiniteElementSpace
        Space indexing the vector entries.

    Returns
    -------
    global_vector : jax.Array
        Assembled vector, shape ``(..., dofs_test, 1)``.

    Notes
    -----
    Contributions from cells sharing a DOF are summed. Leading batch
    dimensions are carried through unchanged.
    """
    expected_shape = (
        test_space.cells_to_dofs.shape[-2],
        1,
        test_space.element.number_of_local_dofs,
        1,
    )
    if local_vectors.shape[-4:] != expected_shape:
        trailing_shape = ", ".join(str(size) for size in expected_shape)
        raise ValueError(
            f"local_vectors must have shape (..., {trailing_shape}), got "
            f"{local_vectors.shape}."
        )

    global_vector = jnp.zeros(
        local_vectors.shape[:-4] + (test_space.number_of_dofs, 1),
        dtype=local_vectors.dtype,
    )
    return global_vector.at[..., test_space.cells_to_dofs, :].add(
        jnp.squeeze(local_vectors, axis=-3)
    )


def gather_vector(global_vector: jax.Array, space: FiniteElementSpace) -> jax.Array:
    """
    Gather a global vector onto the local DOFs of every cell.

    Parameters
    ----------
    global_vector : jax.Array
        Global vector, shape ``(..., dofs_test, 1)``.
    space : FiniteElementSpace
        Space indexing the vector entries.

    Returns
    -------
    local_vectors : jax.Array
        Local vectors, shape ``(..., K, 1, N_test, 1)``.

    Notes
    -----
    This is the adjoint of ``assemble_vector`` and returns the same layout as
    a local vector, so a matrix-free product with local matrices reads
    ``assemble_vector(local_matrices @ gather_vector(x, trial_space),
    test_space)``.
    """
    return jnp.expand_dims(
        jnp.take(global_vector, space.cells_to_dofs, axis=-2), axis=-3
    )


def combine_matrix_blocks(
    blocks: Sequence[Sequence[GlobalMatrix | None]],
) -> GlobalMatrix:
    """
    Combine a grid of sparse blocks into one sparse matrix.

    Parameters
    ----------
    blocks : Sequence[Sequence[GlobalMatrix | None]]
        Rows of blocks; ``None`` stands for a zero block. Every block row and
        block column must contain at least one matrix to fix its size.

    Returns
    -------
    block_matrix : GlobalMatrix
        The combined sparse matrix.

    Notes
    -----
    Block entries are concatenated with their indices offset into the
    combined numbering; nothing is densified or compacted.
    """
    number_of_block_rows = len(blocks)
    number_of_block_columns = len(blocks[0]) if blocks else 0
    if number_of_block_rows == 0 or number_of_block_columns == 0:
        raise ValueError("blocks must contain at least one block.")
    if any(len(row) != number_of_block_columns for row in blocks):
        raise ValueError("Every block row must have the same number of blocks.")

    row_sizes = [
        _block_size(row, axis=0, label=f"block row {r}") for r, row in enumerate(blocks)
    ]
    column_sizes = [
        _block_size(
            [row[c] for row in blocks], axis=1, label=f"block column {c}"
        )
        for c in range(number_of_block_columns)
    ]
    row_offsets = [sum(row_sizes[:r]) for r in range(number_of_block_rows)]
    column_offsets = [sum(column_sizes[:c]) for c in range(number_of_block_columns)]

    data = []
    indices = []
    for r, row in enumerate(blocks):
        for c, block in enumerate(row):
            if block is None:
                continue
            if block.shape != (row_sizes[r], column_sizes[c]):
                raise ValueError(
                    f"Block ({r}, {c}) has shape {block.shape}, expected "
                    f"{(row_sizes[r], column_sizes[c])}."
                )
            offset = jnp.asarray(
                (row_offsets[r], column_offsets[c]), dtype=block.indices.dtype
            )
            data.append(block.data)
            indices.append(block.indices + offset)

    return jax_sparse.BCOO(
        (jnp.concatenate(data, axis=-1), jnp.concatenate(indices, axis=-2)),
        shape=(sum(row_sizes), sum(column_sizes)),
    )


def _block_size(
    blocks: Sequence[GlobalMatrix | None], axis: int, label: str
) -> int:
    """
    Size of a block row or column, taken from its first non-empty block.

    Parameters
    ----------
    blocks : Sequence[GlobalMatrix | None]
        The blocks of one block row or block column.
    axis : int
        ``0`` for a block row (number of rows), ``1`` for a block column.
    label : str
        Name used in the error message.

    Returns
    -------
    size : int
        The shared size along ``axis``.
    """
    for block in blocks:
        if block is not None:
            if block.n_batch != 0 or block.n_dense != 0:
                raise ValueError(
                    "combine_matrix_blocks supports only unbatched 2D BCOO "
                    "blocks without dense dimensions."
                )
            return block.shape[axis]
    raise ValueError(f"{label} contains only None blocks; its size is undefined.")
