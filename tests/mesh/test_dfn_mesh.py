"""Tests for the fracture network mesh ``DfnMesh`` and its constructors."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.mesh import (
    Mesh,
    create_dfn_mesh_from_arrays,
    create_dfn_two_fractures_mesh,
    create_fracture_frames,
)

pytestmark = pytest.mark.unit

# A shared edge from vertex 0 to vertex 1 along the y-axis, and apexes in
# the planes z = 0 (vertices 2, 3, 6) and x = 0 (vertices 4, 5).
_FAN_VERTICES = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (2.0, 0.0, 0.0),
    ]
)
_SHARED_EDGE = (0, 1)


def _fan_mesh(apexes_and_tags, **kwargs):
    cells = [(*_SHARED_EDGE, apex) for apex, _ in apexes_and_tags]
    tags = [tag for _, tag in apexes_and_tags]
    return create_dfn_mesh_from_arrays(_FAN_VERTICES, cells, tags, **kwargs)


def _shared_facet(mesh) -> int:
    facets = np.asarray(mesh.facets_to_vertices)
    return int(np.flatnonzero(np.all(facets == _SHARED_EDGE, axis=-1))[0])


@pytest.mark.parametrize(
    ("apexes_and_tags", "cells_on_edge", "is_trace"),
    [
        (((2, 1), (3, 1)), 2, False),
        (((2, 1), (4, 2)), 2, True),
        (((2, 1), (3, 1), (4, 2)), 3, True),
        (((2, 1), (3, 1), (4, 2), (5, 2)), 4, True),
    ],
    ids=["interior", "l_junction", "t_junction", "x_crossing"],
)
def test_shared_edge_incidence_and_classification(
    apexes_and_tags, cells_on_edge: int, is_trace: bool
) -> None:
    mesh = _fan_mesh(apexes_and_tags)
    facet = _shared_facet(mesh)
    mask = np.asarray(mesh.facet_cell_mask)

    assert mesh.facets_to_cells.shape == (mesh.facets_to_vertices.shape[0], 4)
    assert int(mask[facet].sum()) == cells_on_edge
    assert np.asarray(mesh.facets_to_cells)[facet][mask[facet]].tolist() == list(
        range(cells_on_edge)
    )
    assert (facet in np.asarray(mesh.trace_facets)) == is_trace
    assert facet not in np.asarray(mesh.boundary_facets)
    assert np.all(mask.sum(axis=-1)[np.asarray(mesh.boundary_facets)] == 1)


def test_padded_incidences_point_to_cell_zero() -> None:
    mesh = _fan_mesh(((2, 1), (3, 1), (4, 2)))
    padded = ~np.asarray(mesh.facet_cell_mask)
    assert np.any(padded)
    assert np.all(np.asarray(mesh.facets_to_cells)[padded] == 0)


def test_dfn_mesh_satisfies_mesh_protocol() -> None:
    mesh = _fan_mesh(((2, 1), (4, 2)))
    assert isinstance(mesh, Mesh)
    assert mesh.geometric_dimension == 3
    assert mesh.topological_dimension == 2
    assert mesh.vertex_coordinates.shape == (_FAN_VERTICES.shape[0], 1, 3)


def test_fracture_tags_and_normals() -> None:
    mesh = _fan_mesh(((2, 7), (3, 7), (4, 3)))
    assert mesh.fracture_tags.tolist() == [3, 7]
    assert mesh.cells_to_fractures.tolist() == [1, 1, 0]
    assert mesh.cell_tags.tolist() == [7, 7, 3]
    assert mesh.fracture_normals.shape == (2, 1, 3)
    assert np.allclose(mesh.fracture_normals[:, 0, :], [(1.0, 0.0, 0.0), (0, 0, 1)])


@pytest.mark.parametrize(
    ("apexes_and_tags", "message"),
    [
        (((2, 1), (3, 1), (4, 2), (5, 3)), "more than 2 fractures"),
        (((2, 1), (3, 1), (6, 1), (4, 2)), "more than 2 cells of one fracture"),
        (((2, 1), (4, 1)), "not planar"),
    ],
    ids=["three_fractures", "three_cells_of_one_fracture", "non_planar"],
)
def test_rejects_invalid_networks(apexes_and_tags, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _fan_mesh(apexes_and_tags)


def test_rejects_coincident_vertices() -> None:
    vertices = np.concatenate((_FAN_VERTICES, _FAN_VERTICES[:1]))
    with pytest.raises(ValueError, match="coincident"):
        create_dfn_mesh_from_arrays(vertices, [(0, 1, 2), (7, 1, 4)], [1, 2])


def test_rejects_degenerate_cells() -> None:
    vertices = np.concatenate((_FAN_VERTICES, [(0.0, 2.0, 0.0)]))
    with pytest.raises(ValueError, match="degenerate"):
        create_dfn_mesh_from_arrays(vertices, [(0, 1, 2), (0, 1, 7)], [1, 1])


@pytest.mark.parametrize(
    ("vertices", "cells", "tags", "message"),
    [
        (_FAN_VERTICES[:, :2], [(0, 1, 2)], [1], "vertex_coordinates"),
        (_FAN_VERTICES, [(0, 1)], [1], "cells_to_vertices"),
        (_FAN_VERTICES, [(0, 1, 2)], [1, 2], "cell_tags"),
        (_FAN_VERTICES, [(0, 1, 9)], [1], "outside"),
    ],
    ids=["planar_vertices", "short_cells", "tag_count", "index_range"],
)
def test_rejects_malformed_arrays(vertices, cells, tags, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_dfn_mesh_from_arrays(vertices, cells, tags)


def test_boundary_tags_from_predicates_on_3d_midpoints() -> None:
    mesh = _fan_mesh(
        ((2, 1), (3, 1), (4, 2), (5, 2)),
        boundary_data={
            "fracture_2": lambda midpoints: jnp.abs(midpoints[..., 0, 0]) < 1e-12,
            "fracture_1": lambda midpoints: jnp.ones(midpoints.shape[0], dtype=bool),
        },
    )
    facets = np.asarray(mesh.facets_to_vertices)[np.asarray(mesh.boundary_facets)]
    midpoints = np.mean(np.asarray(mesh.vertex_coordinates)[facets, 0, :], axis=-2)
    expected = np.where(np.isclose(midpoints[..., 0], 0.0), 0, 1)

    assert dict(mesh.boundary_tag_names) == {"fracture_2": 0, "fracture_1": 1}
    assert np.asarray(mesh.boundary_facet_tags).tolist() == expected.tolist()


def test_explicit_boundary_tags() -> None:
    cells = [(0, 1, 2), (0, 1, 4)]
    boundary_pairs = [(0, 2), (1, 2), (0, 4), (1, 4)]
    mesh = create_dfn_mesh_from_arrays(
        _FAN_VERTICES,
        cells,
        [1, 2],
        boundary_facet_tags={
            pair: index % 2 for index, pair in enumerate(boundary_pairs)
        },
        boundary_tag_names={"even": 0, "odd": 1},
    )
    facets = np.asarray(mesh.facets_to_vertices)[np.asarray(mesh.boundary_facets)]
    tags = dict(
        zip(map(tuple, facets.tolist()), mesh.boundary_facet_tags.tolist(), strict=True)
    )
    assert tags == {pair: index % 2 for index, pair in enumerate(boundary_pairs)}


@pytest.mark.parametrize("cells_per_unit_length", [1, 2, 3])
def test_two_fractures_mesh_topology(cells_per_unit_length: int) -> None:
    n = cells_per_unit_length
    mesh = create_dfn_two_fractures_mesh(n)
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]
    incidences = np.asarray(mesh.facet_cell_mask).sum(axis=-1)
    trace_vertices = np.unique(
        np.asarray(mesh.facets_to_vertices)[np.asarray(mesh.trace_facets)]
    )

    assert vertices.shape[0] == 2 * (2 * n + 1) * (n + 1) - (n + 1)
    assert mesh.cells_to_vertices.shape[0] == 8 * n * n
    assert np.bincount(np.asarray(mesh.cells_to_fractures)).tolist() == [4 * n * n] * 2
    assert mesh.trace_facets.shape == (n,)
    assert np.all(incidences[np.asarray(mesh.trace_facets)] == 4)
    assert mesh.boundary_facets.shape == (12 * n,)
    assert np.allclose(vertices[trace_vertices][:, [0, 2]], 0.0)
    assert np.allclose(mesh.fracture_normals[:, 0, :], [(0.0, 0.0, 1.0), (1, 0, 0)])


def test_two_fractures_mesh_rejects_non_positive_resolution() -> None:
    with pytest.raises(ValueError, match="cells_per_unit_length"):
        create_dfn_two_fractures_mesh(0)


def test_fracture_frames_are_orthonormal_tangent_and_right_handed() -> None:
    rng = np.random.default_rng(0)
    triangle = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)])
    vertices, cells, tags = [], [], []
    for fracture in range(5):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        vertices.append(triangle @ rotation.T + rng.normal(size=3))
        cells.append(tuple(range(3 * fracture, 3 * fracture + 3)))
        tags.append(fracture)
    mesh = create_dfn_mesh_from_arrays(np.concatenate(vertices), cells, tags)

    frames = create_fracture_frames(mesh)
    normals = mesh.fracture_normals

    assert frames.shape == (5, 3, 2)
    assert np.allclose(frames.mT @ frames, np.eye(2))
    assert np.allclose(normals @ frames, 0.0)
    determinants = np.linalg.det(np.concatenate((frames, normals.mT), axis=-1))
    assert np.allclose(determinants, 1.0)


_X, _Y, _Z = np.eye(3, dtype=np.int64)


def _trace_endpoint_arc_lengths(mesh, trace: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]
    facets = np.asarray(mesh.trace_facets)[
        np.asarray(mesh.trace_facets_to_traces) == trace
    ]
    endpoints = vertices[np.asarray(mesh.facets_to_vertices)[facets]]
    origin = np.asarray(mesh.trace_origins)[trace, 0]
    direction = np.asarray(mesh.trace_directions)[trace, 0]
    return np.sort((endpoints - origin) @ direction, axis=-1)


def test_two_fractures_mesh_has_one_ordered_trace() -> None:
    mesh = create_dfn_two_fractures_mesh(3)
    arc_lengths = _trace_endpoint_arc_lengths(mesh, 0)

    assert mesh.traces_to_fractures.tolist() == [[0, 1]]
    assert mesh.trace_facets_to_traces.tolist() == [0, 0, 0]
    assert np.allclose(mesh.trace_origins, [[(0.0, 0.0, 0.0)]])
    assert np.allclose(mesh.trace_directions, [[(0.0, 1.0, 0.0)]])
    assert np.allclose(arc_lengths, [[0.0, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 1.0]])


def test_network_without_traces() -> None:
    mesh = _fan_mesh(((2, 1), (3, 1)))
    assert mesh.trace_facets.shape == (0,)
    assert mesh.trace_facets_to_traces.shape == (0,)
    assert mesh.traces_to_fractures.shape == (0, 2)
    assert mesh.trace_origins.shape == (0, 1, 3)
    assert mesh.trace_directions.shape == (0, 1, 3)


def test_traces_of_different_fracture_pairs(lattice_dfn_network_factory) -> None:
    mesh = lattice_dfn_network_factory(
        [
            (1, (-2, 0, 0), _X, _Y, 4, 2),
            (2, (0, 0, -2), _Y, _Z, 2, 4),
            (3, (1, 0, -2), _Y, _Z, 2, 4),
        ],
        2,
    )
    assert mesh.traces_to_fractures.tolist() == [[0, 1], [0, 2]]
    assert mesh.trace_facets_to_traces.tolist() == [0, 0, 1, 1]
    assert np.allclose(mesh.trace_origins[:, 0], [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    for trace in range(2):
        assert np.allclose(
            _trace_endpoint_arc_lengths(mesh, trace), [[0.0, 0.5], [0.5, 1.0]]
        )


def test_disconnected_intersections_are_separate_traces(
    lattice_dfn_network_factory,
) -> None:
    mesh = lattice_dfn_network_factory(
        [
            (1, (-2, 0, 0), _X, _Y, 4, 2),
            (1, (-2, 4, 0), _X, _Y, 4, 2),
            (2, (0, 0, -2), _Y, _Z, 6, 4),
        ],
        2,
    )
    assert mesh.traces_to_fractures.tolist() == [[0, 1], [0, 1]]
    assert mesh.trace_facets_to_traces.tolist() == [0, 0, 1, 1]
    assert np.allclose(mesh.trace_origins[:, 0], [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)])
    assert np.allclose(_trace_endpoint_arc_lengths(mesh, 1), [[0.0, 0.5], [0.5, 1.0]])
