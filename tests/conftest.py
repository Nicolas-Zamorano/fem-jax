"""Shared pytest fixtures for the jax_fem test suite."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.mesh import (
    create_dfn_mesh_from_arrays,
    create_gmsh_unit_square_mesh,
    create_triangle_mesh_from_arrays,
)
from jax_fem.problem import (
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
)
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle

# ---------------------------------------------------------------------------
# Reference element
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_triangle() -> ReferenceTriangle:
    return ReferenceTriangle()


@pytest.fixture
def reference_interval() -> ReferenceInterval:
    return ReferenceInterval()


@pytest.fixture
def p1_element(reference_triangle: ReferenceTriangle) -> LagrangeTriangleP1:
    return LagrangeTriangleP1(reference_cell=reference_triangle)


@pytest.fixture
def p2_element(reference_triangle: ReferenceTriangle) -> LagrangeTriangleP2:
    return LagrangeTriangleP2(reference_cell=reference_triangle)


@pytest.fixture(params=[1, 2, 3, 4, 5], ids=lambda degree: f"degree{degree}")
def quadrature_rule(request, reference_triangle: ReferenceTriangle):
    """A triangle quadrature rule, parameterized over every exactness degree."""
    return reference_triangle.create_quadrature(request.param)


# ---------------------------------------------------------------------------
# Affine physical triangles: reference, translated, rotated, scaled, skew.
# All CCW-oriented and non-degenerate.
# ---------------------------------------------------------------------------

_REFERENCE_TRIANGLE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def _translated_triangle(dx: float, dy: float) -> tuple[tuple[float, float], ...]:
    return tuple((x + dx, y + dy) for x, y in _REFERENCE_TRIANGLE_VERTICES)


def _rotated_triangle(angle_radians: float) -> tuple[tuple[float, float], ...]:
    cos_a, sin_a = math.cos(angle_radians), math.sin(angle_radians)

    def rotate(x: float, y: float) -> tuple[float, float]:
        return (cos_a * x - sin_a * y, sin_a * x + cos_a * y)

    return tuple(rotate(x, y) for x, y in _REFERENCE_TRIANGLE_VERTICES)


def _scaled_triangle(scale: float) -> tuple[tuple[float, float], ...]:
    return tuple((scale * x, scale * y) for x, y in _REFERENCE_TRIANGLE_VERTICES)


_SKEW_TRIANGLE_VERTICES = ((0.3, -0.2), (2.1, 0.4), (0.7, 1.8))

_AFFINE_TRIANGLE_CASES = {
    "reference": _REFERENCE_TRIANGLE_VERTICES,
    "translated": _translated_triangle(2.0, -1.0),
    "rotated": _rotated_triangle(0.7),
    "scaled": _scaled_triangle(2.7),
    "skew": _SKEW_TRIANGLE_VERTICES,
}


@pytest.fixture(
    params=_AFFINE_TRIANGLE_CASES.values(), ids=_AFFINE_TRIANGLE_CASES.keys()
)
def affine_triangle_vertices(request) -> tuple[tuple[float, float], ...]:
    """One affine triangle's vertices: reference, translated, rotated, scaled, skew."""
    return request.param


# ---------------------------------------------------------------------------
# Small meshes
# ---------------------------------------------------------------------------

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_TWO_CELLS = ((0, 1, 2), (0, 2, 3))

_PINWHEEL_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))
_PINWHEEL_CELLS = ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))

_REVERSED_ORIENTATION_VERTICES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
_REVERSED_ORIENTATION_CELLS = ((0, 1, 2), (3, 2, 1))


@pytest.fixture
def two_triangle_unit_square_mesh():
    """The unit square split into 2 triangles by the (1,1)-(0,0) diagonal.

    For topology, continuity, and manual-assembly tests. Every vertex is on
    the boundary, so this mesh has no free P1 DOF.
    """
    return create_triangle_mesh_from_arrays(
        _UNIT_SQUARE_VERTICES, _UNIT_SQUARE_TWO_CELLS
    )


@pytest.fixture
def four_triangle_center_vertex_mesh():
    """The unit square split into 4 triangles fanning from its center.

    Gives exactly one interior (free) P1 DOF at the center vertex; used for
    Dirichlet condensation and PDE-solution tests.
    """
    return create_triangle_mesh_from_arrays(_PINWHEEL_VERTICES, _PINWHEEL_CELLS)


@pytest.fixture
def reversed_orientation_two_triangle_mesh():
    """Two triangles sharing an edge with opposite local vertex orientations.

    ``K_0 = (v0, v1, v2)``, ``K_1 = (v3, v2, v1)`` with ``v0=(0,0)``,
    ``v1=(1,0)``, ``v2=(0,1)``, ``v3=(1,1)``: the shared edge is traversed
    ``v1 -> v2`` in ``K_0`` and ``v2 -> v1`` in ``K_1``.
    """
    return create_triangle_mesh_from_arrays(
        _REVERSED_ORIENTATION_VERTICES, _REVERSED_ORIENTATION_CELLS
    )


@pytest.fixture
def structured_unit_square_mesh_factory():
    """Factory ``n -> TriangleMesh``: a structured unit-square mesh at that resolution.

    For tests needing meshes at several controlled resolutions within one
    test (e.g. correlated convergence-rate studies), rather than one
    resolution per parametrized run. Mesh refinement itself is out of scope
   ; each resolution is a fresh structured mesh.
    """
    return create_gmsh_unit_square_mesh


# ---------------------------------------------------------------------------
# Fracture networks
# ---------------------------------------------------------------------------

_X_AXIS, _Y_AXIS, _Z_AXIS = np.eye(3, dtype=np.int64)


def _lattice_dfn_network(fractures, cells_per_unit_length: int):
    """Network of grid rectangles whose vertices lie on an integer lattice.

    Each fracture is ``(tag, origin, first_axis, second_axis, first_count,
    second_count)`` in lattice units; lattice points shared by several
    fractures become one vertex, and coordinates are lattice points divided
    by ``cells_per_unit_length``.
    """
    lattice_points, cells, tags = [], [], []
    offset = 0
    for tag, origin, first_axis, second_axis, first_count, second_count in fractures:
        first, second = np.meshgrid(
            np.arange(first_count + 1), np.arange(second_count + 1), indexing="ij"
        )
        lattice_points.append(
            np.asarray(origin)
            + np.outer(np.ravel(first), first_axis)
            + np.outer(np.ravel(second), second_axis)
        )
        cell_first, cell_second = np.meshgrid(
            np.arange(first_count), np.arange(second_count), indexing="ij"
        )
        lower_left = offset + np.ravel(cell_first * (second_count + 1) + cell_second)
        lower_right = lower_left + second_count + 1
        cells += [
            np.stack((lower_left, lower_right, lower_right + 1), axis=-1),
            np.stack((lower_left, lower_right + 1, lower_left + 1), axis=-1),
        ]
        tags += [np.full(2 * lower_left.shape[0], tag)]
        offset += lattice_points[-1].shape[0]

    unique_points, inverse = np.unique(
        np.concatenate(lattice_points), axis=0, return_inverse=True
    )
    return create_dfn_mesh_from_arrays(
        unique_points / cells_per_unit_length,
        np.ravel(inverse)[np.concatenate(cells)],
        np.concatenate(tags),
    )


@pytest.fixture
def lattice_dfn_network_factory():
    """Factory ``(fractures, n) -> DfnMesh`` of lattice-aligned grid rectangles.

    Each fracture is ``(tag, origin, first_axis, second_axis, first_count,
    second_count)`` with integer lattice vectors.
    """
    return _lattice_dfn_network


@pytest.fixture
def x_crossing_dfn_mesh():
    """``z = 0`` (tag 1) and ``y = 0`` (tag 2) over ``[-1, 1]^2``, crossing on
    the x-axis, with two cells per unit length; trace facets have four cells.
    """
    return _lattice_dfn_network(
        [
            (1, (-2, -2, 0), _X_AXIS, _Y_AXIS, 4, 4),
            (2, (-2, 0, -2), _X_AXIS, _Z_AXIS, 4, 4),
        ],
        2,
    )


@pytest.fixture
def t_junction_dfn_mesh():
    """``(-1, 1) x (0, 1) x {0}`` (tag 1) and ``{0} x (0, 1) x (0, 1)`` (tag 2),
    the second ending on the first along the y-axis, with two cells per unit
    length; trace facets have three cells.
    """
    return _lattice_dfn_network(
        [
            (1, (-2, 0, 0), _X_AXIS, _Y_AXIS, 4, 2),
            (2, (0, 0, 0), _Y_AXIS, _Z_AXIS, 2, 2),
        ],
        2,
    )


# ---------------------------------------------------------------------------
# Elliptic coefficients
# ---------------------------------------------------------------------------


@pytest.fixture
def isotropic_diffusion():
    """The constant isotropic diffusion tensor ``A = I``."""
    return constant_tensor_coefficient(jnp.eye(2))


@pytest.fixture
def anisotropic_diffusion():
    """A constant, symmetric positive-definite anisotropic diffusion tensor."""
    return constant_tensor_coefficient(jnp.array([[2.0, 0.4], [0.4, 3.0]]))


@pytest.fixture
def zero_advection():
    return constant_vector_coefficient(jnp.zeros(2))


@pytest.fixture
def zero_reaction():
    return constant_scalar_coefficient(0.0)


@pytest.fixture
def zero_source():
    return constant_scalar_coefficient(0.0)
