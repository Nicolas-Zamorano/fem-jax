"""Tests for the in-house Rivara Longest Edge Bisection (LEB) AMR engine.

See ``context/in_house_amr_implementation_plan.md``. ``mark_cells_by_dorfler_bulk_criterion``
is tested Gmsh-free in ``test_amr_marking.py``; this file exercises the pure
``TriangleMesh`` -> ``TriangleMesh`` refinement engine
(``refine_mesh_longest_edge_bisection``) on small, hand-built meshes, with
no external meshing engine involved.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.mesh.amr import (
    _refine_mesh_longest_edge_bisection,
    refine_mesh_longest_edge_bisection,
)

pytestmark = pytest.mark.unit

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))

_TAGGED_UNIT_SQUARE_BOUNDARY_DATA = {
    "bottom": lambda midpoints: jnp.isclose(midpoints[..., 0, 1], 0.0),
    "right": lambda midpoints: jnp.isclose(midpoints[..., 0, 0], 1.0),
    "top": lambda midpoints: jnp.isclose(midpoints[..., 0, 1], 1.0),
    "left": lambda midpoints: jnp.isclose(midpoints[..., 0, 0], 0.0),
}


def _two_triangle_unit_square_mesh():
    return create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)


def _tagged_unit_square_mesh():
    return create_triangle_mesh_from_arrays(
        _UNIT_SQUARE_VERTICES,
        _UNIT_SQUARE_CELLS,
        boundary_data=_TAGGED_UNIT_SQUARE_BOUNDARY_DATA,
    )


def _structured_unit_square_mesh(number_of_cells_per_side: int):
    """A structured N x N unit square mesh, tagged bottom/right/top/left."""
    n = number_of_cells_per_side
    coordinates = np.linspace(0.0, 1.0, n + 1)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="ij")
    vertices = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    def vertex_index(i: int, j: int) -> int:
        return i * (n + 1) + j

    cells = []
    for i, j in itertools.product(range(n), range(n)):
        v00, v10 = vertex_index(i, j), vertex_index(i + 1, j)
        v01, v11 = vertex_index(i, j + 1), vertex_index(i + 1, j + 1)
        cells.append((v00, v10, v11))
        cells.append((v00, v11, v01))

    return create_triangle_mesh_from_arrays(
        vertices, cells, boundary_data=_TAGGED_UNIT_SQUARE_BOUNDARY_DATA
    )


def _min_interior_angle_degrees(mesh) -> float:
    vertices = np.asarray(mesh.vertex_coordinates[:, 0, :])
    cells = np.asarray(mesh.cells_to_vertices)
    triangles = vertices[cells]  # (K, 3, 2)

    min_angle = np.inf
    for triangle in triangles:
        for i in range(3):
            a, b, c = triangle[i], triangle[(i + 1) % 3], triangle[(i + 2) % 3]
            u, v = a - b, c - b
            cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            min_angle = min(min_angle, angle)
    return min_angle


def _facets_have_at_most_two_cells(mesh) -> bool:
    facets_to_cells = np.asarray(mesh.facets_to_cells)
    interior = facets_to_cells[:, 1] != -1
    # Every interior facet must connect exactly 2 distinct cells; every
    # boundary facet exactly 1 -- i.e. zero hanging nodes.
    return bool(np.all(facets_to_cells[interior, 0] != facets_to_cells[interior, 1]))


# ---------------------------------------------------------------------------
# 1. Conformity
# ---------------------------------------------------------------------------


def test_leb_single_cell_refinement_maintains_conformity() -> None:
    mesh = _two_triangle_unit_square_mesh()
    marked = jnp.array([True, False])

    refined = refine_mesh_longest_edge_bisection(mesh, marked)

    assert _facets_have_at_most_two_cells(refined)
    assert refined.cells_to_vertices.shape[0] > mesh.cells_to_vertices.shape[0]


# ---------------------------------------------------------------------------
# 2. No duplicate coincident vertices (shared-midpoint deduplication)
# ---------------------------------------------------------------------------


def test_leb_no_duplicate_coincident_vertices() -> None:
    mesh = _structured_unit_square_mesh(4)
    marked = jnp.zeros(mesh.cells_to_vertices.shape[0], dtype=bool).at[0].set(True)

    refined = refine_mesh_longest_edge_bisection(mesh, marked)

    coordinates = np.asarray(refined.vertex_coordinates[:, 0, :])
    number_of_new_vertices = coordinates.shape[0] - np.asarray(
        mesh.vertex_coordinates
    ).shape[0]

    # No two vertices coincide (would indicate the same edge midpoint was
    # inserted twice instead of shared between matching-bisected cells).
    pairwise_distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )
    np.fill_diagonal(pairwise_distances, np.inf)
    assert np.all(pairwise_distances > 1e-12)
    assert number_of_new_vertices >= 1


# ---------------------------------------------------------------------------
# 3. Boundary tag inheritance
# ---------------------------------------------------------------------------


def test_leb_boundary_tags_preserved() -> None:
    mesh = _structured_unit_square_mesh(4)
    marked = jnp.ones(mesh.cells_to_vertices.shape[0], dtype=bool)

    refined = refine_mesh_longest_edge_bisection(mesh, marked)

    assert refined.boundary_tag_names == mesh.boundary_tag_names
    assert set(np.asarray(refined.boundary_facet_tags).tolist()) == set(
        np.asarray(mesh.boundary_facet_tags).tolist()
    )

    coordinates = np.asarray(refined.vertex_coordinates[:, 0, :])
    facets_to_vertices = np.asarray(refined.facets_to_vertices)
    boundary_facets = np.asarray(refined.boundary_facets)
    boundary_facet_tags = np.asarray(refined.boundary_facet_tags)
    name_to_value = refined.boundary_tag_names

    for facet_index, tag in zip(boundary_facets, boundary_facet_tags, strict=True):
        v0, v1 = facets_to_vertices[facet_index]
        midpoint = (coordinates[v0] + coordinates[v1]) / 2.0
        if tag == name_to_value["bottom"]:
            assert midpoint[1] == pytest.approx(0.0, abs=1e-9)
        elif tag == name_to_value["top"]:
            assert midpoint[1] == pytest.approx(1.0, abs=1e-9)
        elif tag == name_to_value["left"]:
            assert midpoint[0] == pytest.approx(0.0, abs=1e-9)
        elif tag == name_to_value["right"]:
            assert midpoint[0] == pytest.approx(1.0, abs=1e-9)
        else:
            pytest.fail(f"Unexpected boundary tag {tag}.")


# ---------------------------------------------------------------------------
# 4. Physical cell tag inheritance
# ---------------------------------------------------------------------------


def test_leb_cell_tags_preserved() -> None:
    mesh = create_triangle_mesh_from_arrays(
        _UNIT_SQUARE_VERTICES,
        _UNIT_SQUARE_CELLS,
        cell_tags=[3, 7],
    )
    marked = jnp.array([True, False])

    refined = refine_mesh_longest_edge_bisection(mesh, marked)

    assert refined.cell_tags is not None
    tags = np.asarray(refined.cell_tags).tolist()
    # No tag value is invented, and every child inherits its parent's tag:
    # cell 0 (tag 3) was bisected into >= 2 children, all still tag 3 (the
    # shared diagonal is both triangles' longest edge here, so cell 1 gets
    # matching-bisected too -- either way tag 7 must survive somewhere).
    assert set(tags) <= {3, 7}
    assert tags.count(3) >= 2
    assert 7 in tags


# ---------------------------------------------------------------------------
# 5. Propagation stays local
# ---------------------------------------------------------------------------


def test_leb_localized_refinement_decay() -> None:
    small_mesh = _structured_unit_square_mesh(4)
    large_mesh = _structured_unit_square_mesh(40)

    small_marked = jnp.zeros(small_mesh.cells_to_vertices.shape[0], dtype=bool)
    small_marked = small_marked.at[small_marked.shape[0] // 2].set(True)
    large_marked = jnp.zeros(large_mesh.cells_to_vertices.shape[0], dtype=bool)
    large_marked = large_marked.at[large_marked.shape[0] // 2].set(True)

    small_refined = refine_mesh_longest_edge_bisection(small_mesh, small_marked)
    large_refined = refine_mesh_longest_edge_bisection(large_mesh, large_marked)

    small_growth = (
        small_refined.cells_to_vertices.shape[0] - small_mesh.cells_to_vertices.shape[0]
    )
    large_growth = (
        large_refined.cells_to_vertices.shape[0] - large_mesh.cells_to_vertices.shape[0]
    )

    # Marking one interior cell should refine a small, bounded neighborhood
    # regardless of overall mesh size -- propagation decays locally rather
    # than avalanching across the whole domain.
    assert small_growth < 20
    assert large_growth < 20


# ---------------------------------------------------------------------------
# 6. Marked cells are always refined
# ---------------------------------------------------------------------------


def test_leb_marked_cells_are_subset_of_refined_set() -> None:
    mesh = _structured_unit_square_mesh(6)
    marked_indices = [3, 10, 17]
    original_cells = np.asarray(mesh.cells_to_vertices)
    marked_vertex_sets = [
        frozenset(original_cells[i].tolist()) for i in marked_indices
    ]

    marked = jnp.zeros(original_cells.shape[0], dtype=bool)
    marked = marked.at[jnp.array(marked_indices)].set(True)

    refined = refine_mesh_longest_edge_bisection(mesh, marked)
    refined_vertex_sets = {
        frozenset(row.tolist()) for row in np.asarray(refined.cells_to_vertices)
    }

    # Every marked cell's own vertex triple must be gone from the refined
    # mesh -- it was actually split, not left untouched.
    for marked_set in marked_vertex_sets:
        assert marked_set not in refined_vertex_sets


# ---------------------------------------------------------------------------
# 7. Shape regularity
# ---------------------------------------------------------------------------


def test_leb_shape_regularity() -> None:
    mesh = _structured_unit_square_mesh(4)
    initial_min_angle = _min_interior_angle_degrees(mesh)

    rng = np.random.default_rng(0)
    for _ in range(5):
        number_of_cells = mesh.cells_to_vertices.shape[0]
        marked = jnp.asarray(rng.random(number_of_cells) < 0.3)
        if not bool(jnp.any(marked)):
            marked = marked.at[0].set(True)
        mesh = refine_mesh_longest_edge_bisection(mesh, marked)

    final_min_angle = _min_interior_angle_degrees(mesh)

    # Rivara bisection preserves a shape-regularity bound: the minimum
    # angle does not collapse toward 0 as refinement proceeds.
    assert final_min_angle > 0.5 * initial_min_angle
    assert final_min_angle > 10.0


# ---------------------------------------------------------------------------
# 8. Immutability
# ---------------------------------------------------------------------------


def test_leb_immutability() -> None:
    mesh = _structured_unit_square_mesh(4)
    vertex_count_before = mesh.vertex_coordinates.shape[0]
    cell_count_before = mesh.cells_to_vertices.shape[0]
    marked = jnp.ones(cell_count_before, dtype=bool)

    refine_mesh_longest_edge_bisection(mesh, marked)

    assert mesh.vertex_coordinates.shape[0] == vertex_count_before
    assert mesh.cells_to_vertices.shape[0] == cell_count_before


def test_leb_no_marked_cells_returns_input_mesh_unchanged() -> None:
    mesh = _two_triangle_unit_square_mesh()
    marked = jnp.zeros(2, dtype=bool)

    refined = refine_mesh_longest_edge_bisection(mesh, marked)

    assert refined is mesh


# ---------------------------------------------------------------------------
# 9. Defensive termination guard
# ---------------------------------------------------------------------------


def test_leb_defensive_termination_guard() -> None:
    mesh = _structured_unit_square_mesh(4)
    marked = jnp.ones(mesh.cells_to_vertices.shape[0], dtype=bool)

    with pytest.raises(RuntimeError, match="maximum bisection bound"):
        _refine_mesh_longest_edge_bisection(mesh, marked, max_bisections=1)
