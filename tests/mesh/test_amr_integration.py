"""Integration tests for ``adapt_mesh_from_error_estimator``.

See ``context/in_house_amr_implementation_plan.md``. ``mark_cells_by_dorfler_bulk_criterion``
is tested Gmsh-free in ``test_amr_marking.py`` and the pure LEB refinement
engine in ``test_leb_refinement.py``; this file exercises the public,
domain-agnostic entry point end to end on both the Gmsh unit-square and
L-shaped domain factories -- the whole point of replacing Gmsh background
size fields with an in-house LEB engine is that AMR is no longer tied to
one specific domain.
"""

import jax.numpy as jnp
import pytest

from jax_fem.mesh import (
    adapt_mesh_from_error_estimator,
    cell_diameters,
    create_gmsh_l_shaped_mesh,
    create_gmsh_unit_square_mesh,
)

pytestmark = pytest.mark.integration


def _cell_centroid_radii(mesh) -> jnp.ndarray:
    centroids = jnp.mean(mesh.vertex_coordinates[mesh.cells_to_vertices, 0, :], axis=1)
    return jnp.linalg.norm(centroids, axis=-1)


def test_adapt_with_every_cell_marked_grows_the_whole_mesh() -> None:
    """theta=1.0 marks every cell (Dorfler's degenerate/uniform case): every
    cell gets bisected once, an end-to-end sanity check of the marking ->
    LEB pipeline with no localization involved."""
    base = create_gmsh_unit_square_mesh(4)
    base_cell_count = base.cells_to_vertices.shape[0]
    indicators = jnp.ones(base_cell_count)  # uniform, so theta=1 marks all.

    adapted = adapt_mesh_from_error_estimator(base, indicators, theta=1.0)

    assert adapted.cells_to_vertices.shape[0] > base_cell_count
    assert float(jnp.mean(cell_diameters(adapted))) < float(
        jnp.mean(cell_diameters(base))
    )


def test_adapt_localizes_refinement_toward_marked_region() -> None:
    """Indicators concentrated near the re-entrant corner (the origin) mark
    only cells there; the mesh grades toward that region, both in overall
    cell count and in local cell size."""
    base = create_gmsh_l_shaped_mesh(0.4)
    radii = _cell_centroid_radii(base)
    indicators = jnp.where(radii < 0.3, 1.0, 1e-6)  # concentrated at the corner

    adapted = adapt_mesh_from_error_estimator(base, indicators, theta=0.5)

    assert adapted.cells_to_vertices.shape[0] > base.cells_to_vertices.shape[0]

    adapted_radii = _cell_centroid_radii(adapted)
    adapted_diameters = cell_diameters(adapted)
    near_corner_mean = jnp.mean(adapted_diameters[adapted_radii < 0.3])
    far_from_corner_mean = jnp.mean(adapted_diameters[adapted_radii > 0.7])
    assert float(near_corner_mean) < float(far_from_corner_mean)


def test_adapt_is_deterministic() -> None:
    """Same mesh, same indicators -> byte-identical cell count and sizing
    (no hidden randomness in the marking/bisection pipeline)."""
    base = create_gmsh_l_shaped_mesh(0.4)
    radii = _cell_centroid_radii(base)
    indicators = jnp.where(radii < 0.3, 1.0, 1e-6)

    first = adapt_mesh_from_error_estimator(base, indicators, theta=0.5)
    second = adapt_mesh_from_error_estimator(base, indicators, theta=0.5)

    assert first.cells_to_vertices.shape[0] == second.cells_to_vertices.shape[0]
    assert jnp.allclose(cell_diameters(first), cell_diameters(second), atol=1e-12)


def test_adapt_returns_an_independent_mesh_leaving_the_input_untouched() -> None:
    """``TriangleMesh`` immutability: adapting must not mutate ``base``."""
    base = create_gmsh_unit_square_mesh(4)
    base_cell_count_before = base.cells_to_vertices.shape[0]
    indicators = jnp.ones(base_cell_count_before)

    adapt_mesh_from_error_estimator(base, indicators, theta=1.0)

    assert base.cells_to_vertices.shape[0] == base_cell_count_before


def test_adapt_works_identically_across_domains() -> None:
    """Domain-agnostic: the same call works unmodified on both the unit
    square and the L-shaped domain, unlike the old Gmsh-hardwired
    ``adapt_l_shaped_mesh_from_error_estimator``."""
    for base in (create_gmsh_unit_square_mesh(4), create_gmsh_l_shaped_mesh(0.4)):
        indicators = jnp.ones(base.cells_to_vertices.shape[0])
        adapted = adapt_mesh_from_error_estimator(base, indicators, theta=1.0)
        assert adapted.cells_to_vertices.shape[0] > base.cells_to_vertices.shape[0]
