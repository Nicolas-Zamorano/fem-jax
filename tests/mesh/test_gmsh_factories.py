"""Tests for the in-memory Gmsh domain factories and mesh hierarchies.

See Section 2 of the implementation plan (``.context/implementation_plan.md``):
exact integer boundary tagging, ``create_gmsh_unit_square_mesh``,
``create_gmsh_l_shaped_mesh``, and ``create_gmsh_mesh_hierarchy``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.mesh import (
    create_gmsh_l_shaped_mesh,
    create_gmsh_mesh_hierarchy,
    create_gmsh_unit_square_mesh,
    l_shaped_gmsh_geometry,
    unit_square_gmsh_geometry,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# create_gmsh_unit_square_mesh
# ---------------------------------------------------------------------------


def test_unit_square_boundary_tags_are_exact() -> None:
    mesh = create_gmsh_unit_square_mesh(4)
    assert set(mesh.boundary_tag_names) == {"bottom", "right", "top", "left"}

    for tag_name, axis, value in (
        ("bottom", 1, 0.0),
        ("top", 1, 1.0),
        ("left", 0, 0.0),
        ("right", 0, 1.0),
    ):
        tag_value = mesh.boundary_tag_names[tag_name]
        facet_indices = mesh.boundary_facets[mesh.boundary_facet_tags == tag_value]
        assert facet_indices.shape[0] > 0
        facet_vertex_coords = mesh.vertex_coordinates[
            mesh.facets_to_vertices[facet_indices]
        ]
        # Exact equality (not an approximate tolerance check): every matched
        # facet's vertices genuinely lie on the tagged boundary line.
        assert jnp.all(facet_vertex_coords[..., axis] == value)


def test_unit_square_every_boundary_facet_tagged_exactly_once() -> None:
    mesh = create_gmsh_unit_square_mesh(5)
    assert mesh.boundary_facet_tags.shape == mesh.boundary_facets.shape
    assert set(mesh.boundary_facet_tags.tolist()) == {0, 1, 2, 3}
    assert jnp.all(mesh.boundary_facet_tags >= 0)


def test_unit_square_covers_domain_and_is_ccw() -> None:
    mesh = create_gmsh_unit_square_mesh(4)
    assert float(jnp.min(mesh.vertex_coordinates)) == pytest.approx(0.0, abs=1e-12)
    assert float(jnp.max(mesh.vertex_coordinates)) == pytest.approx(1.0, abs=1e-12)

    vertices = np.asarray(mesh.vertex_coordinates[:, 0, :])
    cells = np.asarray(mesh.cells_to_vertices)
    v0, v1, v2 = vertices[cells[:, 0]], vertices[cells[:, 1]], vertices[cells[:, 2]]
    signed_area = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    assert np.all(signed_area > 0.0)
    assert signed_area.sum() == pytest.approx(1.0, abs=1e-10)


def test_unit_square_rejects_non_positive_resolution() -> None:
    with pytest.raises(ValueError, match="number_of_cells_per_side"):
        create_gmsh_unit_square_mesh(0)


# ---------------------------------------------------------------------------
# create_gmsh_l_shaped_mesh
# ---------------------------------------------------------------------------


def test_l_shaped_mesh_has_reentrant_corner_at_origin_and_correct_area() -> None:
    mesh = create_gmsh_l_shaped_mesh(0.3)

    vertices = np.asarray(mesh.vertex_coordinates[:, 0, :])
    cells = np.asarray(mesh.cells_to_vertices)
    v0, v1, v2 = vertices[cells[:, 0]], vertices[cells[:, 1]], vertices[cells[:, 2]]
    signed_area = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    assert np.all(signed_area > 0.0)  # CCW: consistent orientation throughout
    # [-1,1]^2 (area 4) minus the removed [0,1]x[-1,0] quadrant (area 1).
    assert signed_area.sum() == pytest.approx(3.0, abs=1e-8)

    # The origin is a mesh vertex, i.e. the re-entrant corner is resolved
    # exactly rather than only approximated by nearby vertices.
    distances_to_origin = np.linalg.norm(vertices, axis=-1)
    assert np.min(distances_to_origin) < 1e-12


def test_l_shaped_mesh_single_default_boundary_tag() -> None:
    mesh = create_gmsh_l_shaped_mesh(0.3)
    assert set(mesh.boundary_tag_names) == {"boundary"}
    assert set(mesh.boundary_facet_tags.tolist()) == {0}


def test_l_shaped_mesh_rejects_non_positive_target_h() -> None:
    with pytest.raises(ValueError, match="target_h"):
        create_gmsh_l_shaped_mesh(0.0)


# ---------------------------------------------------------------------------
# create_gmsh_mesh_hierarchy
# ---------------------------------------------------------------------------


def test_hierarchy_quadruples_cell_count_each_level() -> None:
    meshes = create_gmsh_mesh_hierarchy(unit_square_gmsh_geometry, 3, levels=2)
    assert len(meshes) == 3

    cell_counts = [mesh.cells_to_vertices.shape[0] for mesh in meshes]
    assert cell_counts[1] == 4 * cell_counts[0]
    assert cell_counts[2] == 4 * cell_counts[1]


def test_hierarchy_preserves_boundary_tags_and_orientation() -> None:
    meshes = create_gmsh_mesh_hierarchy(unit_square_gmsh_geometry, 2, levels=1)
    for mesh in meshes:
        assert set(mesh.boundary_tag_names) == {"bottom", "right", "top", "left"}
        assert set(mesh.boundary_facet_tags.tolist()) == {0, 1, 2, 3}

        vertices = np.asarray(mesh.vertex_coordinates[:, 0, :])
        cells = np.asarray(mesh.cells_to_vertices)
        v0, v1, v2 = vertices[cells[:, 0]], vertices[cells[:, 1]], vertices[cells[:, 2]]
        signed_area = 0.5 * (
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
            - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
        )
        assert np.all(signed_area > 0.0)


def test_hierarchy_works_for_l_shaped_geometry_builder() -> None:
    meshes = create_gmsh_mesh_hierarchy(l_shaped_gmsh_geometry, 0.4, levels=1)
    assert len(meshes) == 2
    assert meshes[1].cells_to_vertices.shape[0] == 4 * meshes[0].cells_to_vertices.shape[0]
    for mesh in meshes:
        assert set(mesh.boundary_tag_names) == {"boundary"}


def test_hierarchy_level_zero_only() -> None:
    meshes = create_gmsh_mesh_hierarchy(unit_square_gmsh_geometry, 3, levels=0)
    assert len(meshes) == 1


def test_hierarchy_rejects_negative_levels() -> None:
    with pytest.raises(ValueError, match="levels"):
        create_gmsh_mesh_hierarchy(unit_square_gmsh_geometry, 3, levels=-1)


def test_hierarchy_composes_inside_an_already_open_session() -> None:
    """create_gmsh_mesh_hierarchy must not finalize a session it did not open."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        meshes = create_gmsh_mesh_hierarchy(unit_square_gmsh_geometry, 2, levels=1)
        assert len(meshes) == 2
        assert gmsh.isInitialized()
    finally:
        gmsh.finalize()
