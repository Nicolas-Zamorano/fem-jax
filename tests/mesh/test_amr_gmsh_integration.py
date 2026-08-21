"""Tests for the Gmsh-touching AMR bridge and public adaptation entry point.

See Section 4.2 of the implementation plan (``.context/implementation_plan.md``,
the AMR stretch item). ``mark_cells_by_dorfler_bulk_criterion`` and
``compute_target_cell_sizes`` are tested Gmsh-free in
``test_amr_marking.py``; this file exercises the ``PostView`` background
size field bridge and ``adapt_l_shaped_mesh_from_error_estimator``.

Note on assertions: Gmsh's 2D mesher grades element size smoothly (a
quality-preserving behavior, not a defect), so a *single* adaptation step
starting from a coarse base mesh only partially reaches an aggressively
localized target size -- full concentration builds up cumulatively across
several solve/estimate/adapt iterations, the way ``adapt_l_shaped_mesh_from_error_estimator``
is meant to be called in a loop. The assertions below check the direction
and existence of that effect, not an exact target-size match.
"""

import jax.numpy as jnp
import pytest

from jax_fem.mesh import (
    adapt_l_shaped_mesh_from_error_estimator,
    cell_diameters,
    create_gmsh_l_shaped_mesh,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _cell_centroid_radii(mesh) -> jnp.ndarray:
    centroids = jnp.mean(mesh.vertex_coordinates[mesh.cells_to_vertices, 0, :], axis=1)
    return jnp.linalg.norm(centroids, axis=-1)


def test_adapt_with_every_cell_marked_shrinks_the_whole_mesh() -> None:
    """theta=1.0 marks every cell (Dorfler's degenerate/uniform case): the
    background field is then a spatially uniform ``h/2`` target, which
    Gmsh's mesher reaches robustly (no localization involved) -- a clean
    end-to-end sanity check of the marking -> sizing -> PostView field ->
    remesh pipeline."""
    base = create_gmsh_l_shaped_mesh(0.4)
    base_cell_count = base.cells_to_vertices.shape[0]
    indicators = jnp.ones(base_cell_count)  # uniform, so theta=1 marks all.

    adapted = adapt_l_shaped_mesh_from_error_estimator(
        base, indicators, theta=1.0, refinement_factor=0.5
    )

    # Halving h roughly quadruples cell count in 2D (same scaling as
    # create_gmsh_mesh_hierarchy's exact 1-to-4 uniform refinement).
    assert adapted.cells_to_vertices.shape[0] > 2 * base_cell_count
    assert float(jnp.mean(cell_diameters(adapted))) < float(
        jnp.mean(cell_diameters(base))
    )


def test_adapt_localizes_refinement_toward_marked_region() -> None:
    """Indicators concentrated near the re-entrant corner (the origin) mark
    only cells there; one adapt step measurably grades the mesh toward that
    region, both in overall cell count and in local cell size."""
    base = create_gmsh_l_shaped_mesh(0.4)
    radii = _cell_centroid_radii(base)
    indicators = jnp.where(radii < 0.3, 1.0, 1e-6)  # concentrated at the corner

    adapted = adapt_l_shaped_mesh_from_error_estimator(
        base, indicators, theta=0.5, refinement_factor=0.5
    )

    assert adapted.cells_to_vertices.shape[0] > base.cells_to_vertices.shape[0]

    adapted_radii = _cell_centroid_radii(adapted)
    adapted_diameters = cell_diameters(adapted)
    near_corner_mean = jnp.mean(adapted_diameters[adapted_radii < 0.3])
    far_from_corner_mean = jnp.mean(adapted_diameters[adapted_radii > 0.7])
    assert float(near_corner_mean) < float(far_from_corner_mean)


def test_adapt_is_deterministic() -> None:
    """Same mesh, same indicators -> byte-identical cell count and sizing
    (no hidden randomness in the marking/sizing/remeshing pipeline)."""
    base = create_gmsh_l_shaped_mesh(0.4)
    radii = _cell_centroid_radii(base)
    indicators = jnp.where(radii < 0.3, 1.0, 1e-6)

    first = adapt_l_shaped_mesh_from_error_estimator(base, indicators, theta=0.5)
    second = adapt_l_shaped_mesh_from_error_estimator(base, indicators, theta=0.5)

    assert first.cells_to_vertices.shape[0] == second.cells_to_vertices.shape[0]
    assert jnp.allclose(cell_diameters(first), cell_diameters(second), atol=1e-12)


def test_adapt_returns_an_independent_mesh_leaving_the_input_untouched() -> None:
    """``TriangleMesh`` immutability: adapting must not mutate ``base``."""
    base = create_gmsh_l_shaped_mesh(0.4)
    base_cell_count_before = base.cells_to_vertices.shape[0]
    indicators = jnp.ones(base_cell_count_before)

    adapt_l_shaped_mesh_from_error_estimator(base, indicators, theta=1.0)

    assert base.cells_to_vertices.shape[0] == base_cell_count_before
