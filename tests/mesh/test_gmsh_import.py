"""Tests for Gmsh model and .msh import paths (Sections 7.3, 7.4)."""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.integration

gmsh = pytest.importorskip("gmsh")

from jax_fem.mesh import (  # noqa: E402
    create_triangle_mesh_from_gmsh_model,
    read_triangle_mesh_from_msh,
)


def _build_tagged_square_model(mesh_size: float = 0.6) -> None:
    """Build a unit square with 4 tagged boundary curves and a tagged surface."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("unit_square")

    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, mesh_size)
    bottom = gmsh.model.geo.addLine(p1, p2)
    right = gmsh.model.geo.addLine(p2, p3)
    top = gmsh.model.geo.addLine(p3, p4)
    left = gmsh.model.geo.addLine(p4, p1)
    loop = gmsh.model.geo.addCurveLoop([bottom, right, top, left])
    surface = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [bottom], name="bottom")
    gmsh.model.addPhysicalGroup(1, [right], name="right")
    gmsh.model.addPhysicalGroup(1, [top], name="top")
    gmsh.model.addPhysicalGroup(1, [left], name="left")
    gmsh.model.addPhysicalGroup(2, [surface], name="domain")

    gmsh.model.mesh.generate(2)


@pytest.fixture
def tagged_square_model():
    _build_tagged_square_model()
    yield gmsh.model
    gmsh.finalize()


def _assert_boundary_edge_on_line(
    mesh, tag_name: str, axis: int, value: float
) -> None:
    tag_value = mesh.boundary_tag_names[tag_name]
    facet_indices = mesh.boundary_facets[mesh.boundary_facet_tags == tag_value]
    assert facet_indices.shape[0] > 0
    facet_vertex_coords = mesh.vertex_coordinates[
        mesh.facets_to_vertices[facet_indices]
    ]
    assert jnp.allclose(facet_vertex_coords[..., axis], value)


def test_boundary_tags_match_physical_curves(tagged_square_model) -> None:
    mesh = create_triangle_mesh_from_gmsh_model(tagged_square_model)

    assert set(mesh.boundary_tag_names) == {"bottom", "right", "top", "left"}
    _assert_boundary_edge_on_line(mesh, "bottom", axis=1, value=0.0)
    _assert_boundary_edge_on_line(mesh, "top", axis=1, value=1.0)
    _assert_boundary_edge_on_line(mesh, "left", axis=0, value=0.0)
    _assert_boundary_edge_on_line(mesh, "right", axis=0, value=1.0)


def test_every_boundary_facet_gets_exactly_one_tag(tagged_square_model) -> None:
    mesh = create_triangle_mesh_from_gmsh_model(tagged_square_model)
    assert mesh.boundary_facet_tags.shape == mesh.boundary_facets.shape
    assert set(mesh.boundary_facet_tags.tolist()) == {0, 1, 2, 3}


def test_cell_tags_from_physical_surface(tagged_square_model) -> None:
    mesh = create_triangle_mesh_from_gmsh_model(tagged_square_model)
    assert mesh.cell_tags is not None
    assert mesh.cell_tags.shape == mesh.cells_to_vertices.shape[:1]
    assert jnp.all(mesh.cell_tags != -1)
    # A single physical surface: every cell shares the same tag value.
    assert len(set(mesh.cell_tags.tolist())) == 1


def test_mesh_covers_the_whole_unit_square(tagged_square_model) -> None:
    mesh = create_triangle_mesh_from_gmsh_model(tagged_square_model)
    assert mesh.geometric_dimension == 2
    assert mesh.topological_dimension == 2
    assert float(jnp.min(mesh.vertex_coordinates)) == pytest.approx(0.0, abs=1e-12)
    assert float(jnp.max(mesh.vertex_coordinates)) == pytest.approx(1.0, abs=1e-12)


def test_read_msh_file_round_trip(tagged_square_model, tmp_path) -> None:
    in_memory_mesh = create_triangle_mesh_from_gmsh_model(tagged_square_model)

    msh_path = tmp_path / "unit_square.msh"
    gmsh.write(str(msh_path))
    gmsh.finalize()  # simulate a fresh process: no live session for the reader

    file_mesh = read_triangle_mesh_from_msh(msh_path)

    assert file_mesh.vertex_coordinates.shape == in_memory_mesh.vertex_coordinates.shape
    assert file_mesh.cells_to_vertices.shape == in_memory_mesh.cells_to_vertices.shape
    assert set(file_mesh.boundary_tag_names) == set(in_memory_mesh.boundary_tag_names)

    # Re-initialize so the fixture's own gmsh.finalize() teardown is a no-op-safe call.
    gmsh.initialize()


def test_reader_reuses_an_already_initialized_session(
    tagged_square_model, tmp_path
) -> None:
    msh_path = tmp_path / "unit_square.msh"
    gmsh.write(str(msh_path))

    assert gmsh.isInitialized()
    mesh = read_triangle_mesh_from_msh(msh_path)
    assert gmsh.isInitialized()  # left open since the fixture owns the session
    assert mesh.geometric_dimension == 2
