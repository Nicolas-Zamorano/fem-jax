"""Tests for TriangleMesh construction from arrays.

See Section 7.3 of the architecture specification. Facet-topology and
boundary-tagging correctness is groundwork the FEM testing plan's DOF/GEO
categories build on, but isn't itself one of the plan's named test IDs.
"""

import jax.numpy as jnp
import pytest

from jax_fem.mesh import create_triangle_mesh_from_arrays

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


@pytest.mark.unit
def test_facet_counts(two_triangle_unit_square_mesh) -> None:
    # 4 boundary edges + 1 shared diagonal = 5 facets total.
    assert two_triangle_unit_square_mesh.facets_to_vertices.shape == (5, 2)
    assert two_triangle_unit_square_mesh.boundary_facets.shape == (4,)


@pytest.mark.unit
def test_interior_facet_has_two_cells(two_triangle_unit_square_mesh) -> None:
    facets_to_cells = two_triangle_unit_square_mesh.facets_to_cells
    interior_mask = jnp.all(facets_to_cells != -1, axis=1)
    assert int(jnp.sum(interior_mask)) == 1
    (interior_facet_index,) = jnp.nonzero(interior_mask)[0]
    interior_cells = set(facets_to_cells[interior_facet_index].tolist())
    assert interior_cells == {0, 1}
    # The shared diagonal connects vertices 0 and 2.
    assert set(
        two_triangle_unit_square_mesh.facets_to_vertices[interior_facet_index].tolist()
    ) == {0, 2}


@pytest.mark.unit
def test_boundary_facets_have_one_cell(two_triangle_unit_square_mesh) -> None:
    boundary_facets_to_cells = two_triangle_unit_square_mesh.facets_to_cells[
        two_triangle_unit_square_mesh.boundary_facets
    ]
    assert jnp.all(boundary_facets_to_cells[:, 1] == -1)
    assert jnp.all(boundary_facets_to_cells[:, 0] != -1)


@pytest.mark.unit
def test_cells_to_facets_consistent_with_facets_to_cells(
    two_triangle_unit_square_mesh,
) -> None:
    cells_to_facets = two_triangle_unit_square_mesh.cells_to_facets
    facets_to_cells = two_triangle_unit_square_mesh.facets_to_cells
    for cell_index in range(cells_to_facets.shape[0]):
        for facet_index in cells_to_facets[cell_index].tolist():
            assert cell_index in facets_to_cells[facet_index].tolist()


@pytest.mark.unit
def test_default_boundary_tag(two_triangle_unit_square_mesh) -> None:
    assert two_triangle_unit_square_mesh.boundary_tag_names == {"boundary": 0}
    assert jnp.all(two_triangle_unit_square_mesh.boundary_facet_tags == 0)


@pytest.mark.unit
def test_predicate_boundary_tagging() -> None:
    mesh = create_triangle_mesh_from_arrays(
        _UNIT_SQUARE_VERTICES,
        _UNIT_SQUARE_CELLS,
        boundary_data={
            "bottom": lambda x: jnp.isclose(x[..., 0, 1], 0.0),
            "right": lambda x: jnp.isclose(x[..., 0, 0], 1.0),
            "top": lambda x: jnp.isclose(x[..., 0, 1], 1.0),
            "left": lambda x: jnp.isclose(x[..., 0, 0], 0.0),
        },
    )
    assert mesh.boundary_tag_names == {"bottom": 0, "right": 1, "top": 2, "left": 3}
    # Each of the 4 boundary edges should get a distinct tag.
    assert set(mesh.boundary_facet_tags.tolist()) == {0, 1, 2, 3}


@pytest.mark.unit
def test_unmatched_boundary_facet_raises() -> None:
    with pytest.raises(ValueError, match="matched no predicate"):
        create_triangle_mesh_from_arrays(
            _UNIT_SQUARE_VERTICES,
            _UNIT_SQUARE_CELLS,
            boundary_data={"bottom": lambda x: jnp.isclose(x[..., 0, 1], 0.0)},
        )


@pytest.mark.unit
def test_first_matching_predicate_wins() -> None:
    mesh = create_triangle_mesh_from_arrays(
        _UNIT_SQUARE_VERTICES,
        _UNIT_SQUARE_CELLS,
        boundary_data={
            "everything": lambda x: jnp.ones(x.shape[0], dtype=bool),
            "bottom": lambda x: jnp.isclose(x[..., 0, 1], 0.0),
        },
    )
    assert set(mesh.boundary_facet_tags.tolist()) == {0}


@pytest.mark.unit
def test_out_of_range_vertex_index_raises() -> None:
    with pytest.raises(ValueError, match="vertex indices"):
        create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, ((0, 1, 4),))


@pytest.mark.unit
def test_bad_vertex_shape_raises() -> None:
    with pytest.raises(ValueError, match="vertex_coordinates"):
        create_triangle_mesh_from_arrays(((0.0, 0.0, 0.0),), ((0, 0, 0),))


@pytest.mark.unit
def test_non_manifold_facet_raises() -> None:
    # Three cells all sharing the (0, 1) edge: a non-manifold configuration.
    vertices = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, -1.0), (0.5, 2.0))
    cells = ((0, 1, 2), (0, 1, 3), (0, 1, 4))
    with pytest.raises(ValueError, match="Non-manifold"):
        create_triangle_mesh_from_arrays(vertices, cells)


@pytest.mark.unit
def test_geometric_and_topological_dimension(two_triangle_unit_square_mesh) -> None:
    assert two_triangle_unit_square_mesh.geometric_dimension == 2
    assert two_triangle_unit_square_mesh.topological_dimension == 2
