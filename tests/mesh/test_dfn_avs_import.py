"""Tests for reading fracture network meshes from AVS UCD files.

``data/dfngen_two_fractures_full_mesh.inp`` is the ``full_mesh.inp`` written
by dfnWorks (``pydfnworks``) for the two-fracture benchmark network, shifted
to ``y`` in ``(-1/2, 1/2)`` because dfnWorks centres the domain at the
origin: domain size ``(2, 1, 2)``, ``h = 0.5``, two user rectangles of
radius 1.5 through the origin with normals ``(0, 0, 1)`` and ``(1, 0, 0)``,
and ``mesh_network(uniform_mesh=True)``.
"""

from pathlib import Path

import numpy as np
import pytest

from jax_fem.mesh import read_dfn_mesh_avs

pytestmark = pytest.mark.unit

DFNGEN_MESH = Path(__file__).parents[1] / "data" / "dfngen_two_fractures_full_mesh.inp"

_SMALL_AVS = """\
        5          3          2          1          0
10   0.0 0.0 0.0
20   0.0 1.0 0.0
30   1.0 0.0 0.0
40  -1.0 0.0 0.0
50   0.0 0.0 1.0
1        4   tri 10 20 30
2        4   tri 20 10 40
3        9   tri 10 20 50
00002  1  1
imt1, integer
dfield, real
    1  4  0.1E+01
    2  4  0.1E+01
    3  4  0.1E+01
    4  4  0.1E+01
    5  9  0.1E+01
00001  1
xnorm, real
    1  0.0
    2  0.0
    3  1.0
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "full_mesh.inp"
    path.write_text(text)
    return path


def _edge_lengths(mesh, facets) -> np.ndarray:
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]
    pairs = np.asarray(mesh.facets_to_vertices)[np.asarray(facets)]
    return np.linalg.norm(vertices[pairs[..., 1]] - vertices[pairs[..., 0]], axis=-1)


def test_reads_nodes_cells_and_materials(tmp_path: Path) -> None:
    mesh = read_dfn_mesh_avs(_write(tmp_path, _SMALL_AVS))

    assert np.asarray(mesh.vertex_coordinates)[:, 0, :].tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert mesh.cells_to_vertices.tolist() == [[0, 1, 2], [1, 0, 3], [0, 1, 4]]
    assert mesh.cell_tags.tolist() == [4, 4, 9]
    assert mesh.fracture_tags.tolist() == [4, 9]
    assert mesh.trace_facets.shape == (1,)


def test_passes_boundary_tagging_through(tmp_path: Path) -> None:
    mesh = read_dfn_mesh_avs(
        _write(tmp_path, _SMALL_AVS),
        boundary_data={"anything": lambda midpoints: midpoints[..., 0, 1] > -1.0},
    )
    assert dict(mesh.boundary_tag_names) == {"anything": 0}
    assert mesh.boundary_facets.shape == (6,)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("tri 10 20 50", "quad 10 20 50 30", "not a triangle"),
        ("tri 10 20 50", "tri 10 20 60", "unknown node ids"),
    ],
    ids=["quad_cell", "unknown_node"],
)
def test_rejects_malformed_files(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_dfn_mesh_avs(_write(tmp_path, _SMALL_AVS.replace(old, new)))


def test_rejects_truncated_file(tmp_path: Path) -> None:
    truncated = "\n".join(_SMALL_AVS.splitlines()[:7])
    with pytest.raises(ValueError, match="expected 5 node lines and 3 cell lines"):
        read_dfn_mesh_avs(_write(tmp_path, truncated))


@pytest.mark.integration
def test_dfngen_mesh_geometry_and_traces() -> None:
    mesh = read_dfn_mesh_avs(DFNGEN_MESH)
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]
    corners = vertices[np.asarray(mesh.cells_to_vertices)]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=-1
    )
    incidences = np.asarray(mesh.facet_cell_mask).sum(axis=-1)

    assert vertices.shape == (60, 3)
    assert mesh.cells_to_vertices.shape == (80, 3)
    assert mesh.fracture_tags.tolist() == [1, 2]
    assert areas.sum() == pytest.approx(4.0, rel=1e-12)
    assert np.all(incidences[np.asarray(mesh.trace_facets)] == 4)
    assert _edge_lengths(mesh, mesh.trace_facets).sum() == pytest.approx(1.0)
    assert _edge_lengths(mesh, mesh.boundary_facets).sum() == pytest.approx(12.0)
    assert np.allclose(mesh.fracture_normals[:, 0, :], [(0.0, 0.0, 1.0), (1, 0, 0)])
