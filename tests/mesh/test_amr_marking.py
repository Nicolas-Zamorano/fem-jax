"""Tests for the pure (Gmsh-free) AMR marking and sizing layer.

See Section 4.2 of the implementation plan (``.context/implementation_plan.md``,
the AMR stretch item): ``mark_cells_by_dorfler_bulk_criterion`` and
``compute_target_cell_sizes`` are ordinary JAX functions, unit-testable
without Gmsh at all -- the Gmsh-only bridge is exercised separately in
``test_amr_gmsh_integration.py``.
"""

import jax.numpy as jnp
import pytest

from jax_fem.mesh import (
    cell_diameters,
    compute_target_cell_sizes,
    create_triangle_mesh_from_arrays,
    mark_cells_by_dorfler_bulk_criterion,
)

pytestmark = pytest.mark.unit

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


# ---------------------------------------------------------------------------
# mark_cells_by_dorfler_bulk_criterion
# ---------------------------------------------------------------------------


def test_dorfler_marking_hand_derived_case() -> None:
    """Squared indicators [9, 1, 1, 1], total 12, theta=0.5 -> target 6.

    Sorted descending [9, 1, 1, 1], cumulative [9, 10, 11, 12]: the smallest
    prefix reaching >= 6 is just the first cell.
    """
    indicators = jnp.array([3.0, 1.0, 1.0, 1.0])
    marked = mark_cells_by_dorfler_bulk_criterion(indicators, theta=0.5)
    assert marked.tolist() == [True, False, False, False]


def test_dorfler_marking_theta_one_marks_every_cell() -> None:
    indicators = jnp.array([3.0, 1.0, 1.0, 1.0])
    marked = mark_cells_by_dorfler_bulk_criterion(indicators, theta=1.0)
    assert bool(jnp.all(marked))


def test_dorfler_marking_is_permutation_invariant() -> None:
    """ERR/ASM-style: marking depends only on the multiset of indicators,
    not on their order -- the same physical cells get marked regardless of
    how they're numbered."""
    indicators = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
    permutation = jnp.array([4, 1, 3, 0, 2])
    marked = mark_cells_by_dorfler_bulk_criterion(indicators, theta=0.6)
    marked_permuted = mark_cells_by_dorfler_bulk_criterion(
        indicators[permutation], theta=0.6
    )
    assert marked[permutation].tolist() == marked_permuted.tolist()


@pytest.mark.parametrize("theta", [0.0, -0.1, 1.1])
def test_dorfler_marking_rejects_theta_outside_unit_interval(theta: float) -> None:
    with pytest.raises(ValueError, match="theta"):
        mark_cells_by_dorfler_bulk_criterion(jnp.array([1.0, 2.0]), theta=theta)


def test_dorfler_marking_marks_at_least_the_largest_indicator() -> None:
    """A vacuous empty marking would defeat the whole point of AMR."""
    indicators = jnp.array([0.1, 0.2, 5.0, 0.3])
    marked = mark_cells_by_dorfler_bulk_criterion(indicators, theta=0.1)
    assert bool(jnp.sum(marked)) >= 1
    assert bool(marked[2])  # the largest indicator is always marked


# ---------------------------------------------------------------------------
# compute_target_cell_sizes
# ---------------------------------------------------------------------------


def test_target_sizes_halve_marked_cells_and_preserve_unmarked() -> None:
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    current = cell_diameters(mesh)  # both cells: hypotenuse sqrt(2)

    marked = jnp.array([True, False])
    target = compute_target_cell_sizes(mesh, marked, refinement_factor=0.5)

    assert target[0] == pytest.approx(float(current[0]) * 0.5, abs=1e-12)
    assert target[1] == pytest.approx(float(current[1]), abs=1e-12)


def test_target_sizes_no_marked_cells_is_identity() -> None:
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    current = cell_diameters(mesh)
    marked = jnp.zeros(2, dtype=bool)
    target = compute_target_cell_sizes(mesh, marked, refinement_factor=0.3)
    assert jnp.allclose(target, current, atol=1e-12)


@pytest.mark.parametrize("refinement_factor", [0.0, 1.0, -0.5, 1.5])
def test_target_sizes_rejects_refinement_factor_outside_open_unit_interval(
    refinement_factor: float,
) -> None:
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    with pytest.raises(ValueError, match="refinement_factor"):
        compute_target_cell_sizes(
            mesh, jnp.array([True, False]), refinement_factor=refinement_factor
        )
