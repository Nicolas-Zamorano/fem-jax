"""Shared pytest fixtures for the jax_fem test suite.

See Section 3 of the FEM testing plan (``.claude/FEM_Testing_Plan.md``).
"""

import math

import jax.numpy as jnp
import pytest

from jax_fem.element import LagrangeTriangleP1
from jax_fem.mesh import (
    create_structured_unit_square_mesh,
    create_triangle_mesh_from_arrays,
)
from jax_fem.problem import (
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
)
from jax_fem.reference_cell import ReferenceTriangle

# ---------------------------------------------------------------------------
# Reference element
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_triangle() -> ReferenceTriangle:
    return ReferenceTriangle()


@pytest.fixture
def p1_element(reference_triangle: ReferenceTriangle) -> LagrangeTriangleP1:
    return LagrangeTriangleP1(reference_cell=reference_triangle)


@pytest.fixture(params=[1, 2, 3, 4, 5], ids=lambda degree: f"degree{degree}")
def quadrature_rule(request, reference_triangle: ReferenceTriangle):
    """A triangle quadrature rule, parameterized over every exactness degree."""
    return reference_triangle.create_quadrature(request.param)


# ---------------------------------------------------------------------------
# Affine physical triangles: reference, translated, rotated, scaled, skew.
# All CCW-oriented and non-degenerate (Section 3).
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
    Dirichlet condensation and PDE-solution tests (SOL-01).
    """
    return create_triangle_mesh_from_arrays(_PINWHEEL_VERTICES, _PINWHEEL_CELLS)


@pytest.fixture
def reversed_orientation_two_triangle_mesh():
    """Two triangles sharing an edge with opposite local vertex orientations.

    ``K_0 = (v0, v1, v2)``, ``K_1 = (v3, v2, v1)`` with ``v0=(0,0)``,
    ``v1=(1,0)``, ``v2=(0,1)``, ``v3=(1,1)``: the shared edge is traversed
    ``v1 -> v2`` in ``K_0`` and ``v2 -> v1`` in ``K_1`` (DOF-02, DOF-04).
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
    (Section 2.1); each resolution is a fresh structured mesh.
    """
    return create_structured_unit_square_mesh


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
