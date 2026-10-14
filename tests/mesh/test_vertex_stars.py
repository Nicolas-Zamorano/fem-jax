"""Tests for the vertex-star (patch) topology of ``TriangleMesh``."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.mesh import (
    create_triangle_mesh_from_arrays,
    refine_mesh_longest_edge_bisection,
)


def _structured_unit_square(cells_per_side: int):
    ticks = np.linspace(0.0, 1.0, cells_per_side + 1)
    vertices = np.array([(x, y) for y in ticks for x in ticks])
    cells = []
    for j in range(cells_per_side):
        for i in range(cells_per_side):
            a = j * (cells_per_side + 1) + i
            diagonal = a + cells_per_side + 2
            cells += [(a, a + 1, diagonal), (a, diagonal, diagonal - 1)]
    return create_triangle_mesh_from_arrays(vertices, np.array(cells))


def _assert_stars_match_brute_force(mesh) -> None:
    cells_to_vertices = np.asarray(mesh.cells_to_vertices)
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]
    vertices_to_cells = np.asarray(mesh.vertices_to_cells)
    local_vertices = np.asarray(mesh.vertices_to_local_vertex)
    mask = np.asarray(mesh.vertex_patch_mask)
    diameters = np.asarray(mesh.vertex_patch_diameters)

    for vertex in range(vertices.shape[0]):
        expected_cells = np.flatnonzero(np.any(cells_to_vertices == vertex, axis=1))
        star_cells = vertices_to_cells[vertex][mask[vertex]]
        assert sorted(star_cells.tolist()) == expected_cells.tolist()
        star_local_vertices = local_vertices[vertex][mask[vertex]]
        assert np.all(cells_to_vertices[star_cells, star_local_vertices] == vertex)
        star_points = vertices[np.unique(cells_to_vertices[expected_cells])]
        pairwise = np.linalg.norm(star_points[:, None] - star_points[None], axis=-1)
        assert diameters[vertex] == pytest.approx(pairwise.max(), rel=1e-14)


@pytest.mark.unit
def test_vertex_star_valences_on_structured_mesh() -> None:
    cells_per_side = 4
    mesh = _structured_unit_square(cells_per_side)
    valences = np.asarray(jnp.sum(mesh.vertex_patch_mask, axis=-1))
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]

    on_boundary = np.any(np.isclose(vertices, 0.0) | np.isclose(vertices, 1.0), axis=1)
    is_corner = np.all(np.isclose(vertices, 0.0) | np.isclose(vertices, 1.0), axis=1)

    assert mesh.vertices_to_cells.shape == (vertices.shape[0], 6)
    assert np.all(valences[~on_boundary] == 6)
    assert sorted(valences[is_corner].tolist()) == [1, 1, 2, 2]
    assert np.all(valences[on_boundary & ~is_corner] == 3)


@pytest.mark.unit
def test_vertex_stars_match_brute_force_on_structured_mesh() -> None:
    _assert_stars_match_brute_force(_structured_unit_square(3))


@pytest.mark.unit
def test_vertex_patch_diameters_on_structured_mesh() -> None:
    cells_per_side = 4
    spacing = 1.0 / cells_per_side
    mesh = _structured_unit_square(cells_per_side)
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]
    interior = np.all((vertices > 0.0) & (vertices < 1.0), axis=1)

    # An interior star spans the two diagonal neighbours (-s, -s) and (s, s).
    expected = 2.0 * np.sqrt(2.0) * spacing
    assert np.allclose(np.asarray(mesh.vertex_patch_diameters)[interior], expected)


@pytest.mark.unit
def test_padded_slots_point_to_cell_zero() -> None:
    mesh = _structured_unit_square(3)
    padded = ~np.asarray(mesh.vertex_patch_mask)
    assert np.any(padded)
    assert np.all(np.asarray(mesh.vertices_to_cells)[padded] == 0)
    assert np.all(np.asarray(mesh.vertices_to_local_vertex)[padded] == 0)


@pytest.mark.unit
@pytest.mark.integration
def test_vertex_stars_are_rebuilt_after_refinement() -> None:
    mesh = _structured_unit_square(3)
    marked = jnp.zeros(mesh.cells_to_vertices.shape[0], dtype=bool).at[:4].set(True)
    refined = refine_mesh_longest_edge_bisection(mesh, marked)

    assert refined.cells_to_vertices.shape[0] > mesh.cells_to_vertices.shape[0]
    _assert_stars_match_brute_force(refined)
