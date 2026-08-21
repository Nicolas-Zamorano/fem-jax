"""
Batched geometry and basis evaluation on one quadrature rule.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jax_fem._shapes import check_shape
from jax_fem.reference_cell.quadrature import QuadratureRule
from jax_fem.space.finite_element_space import FiniteElementSpace

_VALID_MAPPINGS = ("identity", "contravariant_piola")


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class CellBasis:
    """
    Geometry and basis evaluation of a finite element space at one quadrature rule.

    Quantities that are mathematically constant along the cell axis or the
    quadrature axis (true for any affine, straight-sided triangle) are
    stored at their natural, un-repeated shape rather than broadcast out to
    the full ``(K, Q, ...)`` shape: they broadcast correctly wherever they
    are actually used (e.g. multiplied against the physical measure or
    against a genuinely ``(K, Q, ...)``-shaped coefficient).

    Two element mapping families are supported (``space.element.mapping``,
    Section 5.2), each populating a different pair of ``gradients`` /
    ``divergences``:

    - ``"identity"`` (every scalar Lagrange element: P1, P2, P0): ``values``
      is cell-independent (collapsed to ``K = 1``); ``gradients`` is the
      covariant-transformed physical gradient; ``divergences`` is ``None``.
    - ``"contravariant_piola"`` (RT0): ``values`` genuinely varies per cell
      (the Piola map depends on the cell's own Jacobian, unlike a scalar
      element's reference values); ``divergences`` is the physical
      divergence; ``gradients`` is ``None`` (a vector field's relevant
      differential quantity here is its divergence, not a gradient).

    Attributes
    ----------
    space : FiniteElementSpace
        The finite element space being evaluated.
    quadrature : QuadratureRule
        The quadrature rule the space is evaluated at.
    physical_points : jax.Array
        Quadrature points mapped to physical coordinates, shape
        ``(K, Q, 1, d)``.
    jacobians : jax.Array
        Affine geometry map Jacobians, ``d(physical)/d(reference)``, shape
        ``(K, 1, d, d)``: constant in ``Q`` for straight-sided triangles.
    inverse_jacobians : jax.Array
        Shape ``(K, 1, d, d)``. Inverse of the affine geometry map Jacobians.
    jacobian_determinants : jax.Array
        Shape ``(K, 1, 1, 1)``. Determinant of the affine geometry map Jacobians.
    physical_weights : jax.Array
        Shape ``(K, Q, 1, 1)``. Physical weights for the quadrature rule.
    values : jax.Array
        Basis values at the quadrature points. For an ``"identity"``-mapped
        element, shape ``(1, Q, N_phi, value_dim)`` (constant in ``K``: a
        scalar Lagrange element's reference values do not depend on which
        physical cell they are evaluated on). For a
        ``"contravariant_piola"``-mapped element, shape
        ``(K, Q, N_phi, value_dim)`` (genuinely cell-dependent), already
        sign-corrected (``space.local_dof_signs``).
    gradients : jax.Array | None
        Basis gradients with respect to physical coordinates, shape
        ``(K, 1, N_phi, d)`` if the element's reference gradients are
        constant (e.g. P1, exploiting straight-sided triangles), or
        ``(K, Q, N_phi, d)`` if they vary with the quadrature point (e.g.
        P2). ``None`` for a ``"contravariant_piola"``-mapped element.
    divergences : jax.Array | None
        Physical divergence of each (vector-valued) basis function, shape
        ``(K, 1, N_phi, 1)`` or ``(K, Q, N_phi, 1)`` (same constant-vs-
        varying duality as ``gradients``), already sign-corrected. ``None``
        for an ``"identity"``-mapped element.
    """

    space: FiniteElementSpace
    quadrature: QuadratureRule
    physical_points: jax.Array
    jacobians: jax.Array
    inverse_jacobians: jax.Array
    jacobian_determinants: jax.Array
    physical_weights: jax.Array
    values: jax.Array
    gradients: jax.Array | None
    divergences: jax.Array | None

    def __post_init__(self) -> None:
        number_of_cells = self.space.cells_to_dofs.shape[0]
        number_of_points = self.quadrature.points.shape[1]
        number_of_local_dofs = self.space.element.number_of_local_dofs
        dimension = self.space.mesh.geometric_dimension
        value_dim = self.space.element.value_shape[0]
        mapping = self.space.element.mapping

        check_shape(
            self.physical_points,
            (number_of_cells, number_of_points, 1, dimension),
            "physical_points",
        )
        check_shape(
            self.jacobians, (number_of_cells, 1, dimension, dimension), "jacobians"
        )
        check_shape(
            self.inverse_jacobians,
            (number_of_cells, 1, dimension, dimension),
            "inverse_jacobians",
        )
        check_shape(
            self.jacobian_determinants,
            (number_of_cells, 1, 1, 1),
            "jacobian_determinants",
        )
        check_shape(
            self.physical_weights,
            (number_of_cells, number_of_points, 1, 1),
            "physical_weights",
        )

        if mapping not in _VALID_MAPPINGS:
            raise ValueError(
                f"Unsupported element.mapping {mapping!r}; expected one of "
                f"{_VALID_MAPPINGS}."
            )

        if mapping == "identity":
            check_shape(
                self.values,
                (1, number_of_points, number_of_local_dofs, value_dim),
                "values",
            )
            _check_constant_or_varying(
                self.gradients,
                "gradients",
                (number_of_cells, 1, number_of_local_dofs, dimension),
                (number_of_cells, number_of_points, number_of_local_dofs, dimension),
            )
            if self.divergences is not None:
                raise ValueError(
                    "divergences must be None for an 'identity'-mapped element."
                )
        else:  # "contravariant_piola"
            check_shape(
                self.values,
                (number_of_cells, number_of_points, number_of_local_dofs, value_dim),
                "values",
            )
            _check_constant_or_varying(
                self.divergences,
                "divergences",
                (number_of_cells, 1, number_of_local_dofs, 1),
                (number_of_cells, number_of_points, number_of_local_dofs, 1),
            )
            if self.gradients is not None:
                raise ValueError(
                    "gradients must be None for a 'contravariant_piola'-mapped "
                    "element."
                )


def _check_constant_or_varying(
    array: jax.Array | None,
    name: str,
    constant_shape: tuple[int, ...],
    varying_shape: tuple[int, ...],
) -> None:
    """Require ``array`` (required, not None) to have either shape."""
    if array is None or array.shape not in (constant_shape, varying_shape):
        actual = None if array is None else array.shape
        raise ValueError(
            f"{name} must have shape {constant_shape} (constant reference "
            f"values) or {varying_shape} (quadrature-point-varying reference "
            f"values), got {actual}."
        )


def create_cell_basis(
    space: FiniteElementSpace, quadrature: QuadratureRule
) -> CellBasis:
    """
    Evaluate a finite element space's geometry and basis at one quadrature rule.

    The affine geometry map is built from the mesh's corner vertices
    (``mesh.cells_to_vertices``), independent of the element's DOF map
    (``space.cells_to_dofs``). The two coincide for P1, but must not be
    conflated: the geometry mapping stays affine for straight-sided
    triangles even for higher-order or vector-valued elements whose DOFs
    are not mesh vertices.

    Dispatches on ``space.element.mapping`` (Section 5.2): an
    ``"identity"``-mapped (scalar Lagrange) element's physical values equal
    its reference values, with gradients transformed covariantly
    (``grad_ref @ J^-1``); a ``"contravariant_piola"``-mapped (RT0) element's
    physical values and divergences use the Piola map
    (``value = (1/det(J)) * ref_value @ J^T``, ``div = (1/det(J)) *
    ref_div``), each additionally scaled by ``space.local_dof_signs`` so a
    shared facet's global DOF has one unambiguous orientation.

    Parameters
    ----------
    space: FiniteElementSpace
        The finite element space to evaluate.
    quadrature: QuadratureRule
        The quadrature rule to evaluate at, on a reference cell compatible
        with ``space.element.reference_cell``.

    Returns
    -------
    cell_basis : CellBasis
        The cell basis evaluated at the quadrature points.
    """
    mesh = space.mesh
    element = space.element
    reference_cell = quadrature.reference_cell

    # (K, 3, 1, 2): physical coordinates of each cell's 3 corner vertices.
    cell_vertex_coordinates = mesh.vertex_coordinates[mesh.cells_to_vertices]
    (
        cell_vertex_1_coordinates,
        cell_vertex_2_coordinates,
        cell_vertex_3_coodinates
    ) = jnp.split(cell_vertex_coordinates, 3, axis = -3)
    edge_1 = cell_vertex_2_coordinates - cell_vertex_1_coordinates
    edge_2 = cell_vertex_3_coodinates - cell_vertex_1_coordinates
    # Columns of the Jacobian are the physical images of the reference
    # triangle's two non-origin vertices: d(physical)/d(reference).
    # concatenate(axis=-2) stacks edge_1/edge_2 as rows; .mT makes them
    # columns instead, matching the row-vector convention used below
    # (physical = origin + reference @ J^T, grad_phys = grad_ref @ J^-1).
    jacobians = jnp.concatenate((edge_1, edge_2), axis=-2).mT  # (K, 1, 2, 2)
    inverse_jacobians = jnp.linalg.inv(jacobians)
    jacobian_determinants = jnp.expand_dims(jnp.linalg.det(jacobians), axis =(-1,-2))

    physical_weights = (quadrature.weights
        * reference_cell.measure
        * jacobian_determinants)

    # Row-vector points: x_row = origin_row + xhat_row @ J^T.
    physical_points = cell_vertex_1_coordinates + quadrature.points @ jacobians.mT

    reference_values = element.tabulate_basis_values(quadrature.points)

    if element.mapping == "identity":
        values = reference_values
        reference_gradients = element.tabulate_basis_gradients(quadrature.points)
        # Physical gradients via the chain rule, row-vector convention:
        # grad_x = grad_xhat @ J^{-1}.
        gradients = reference_gradients @ inverse_jacobians
        divergences = None
    elif element.mapping == "contravariant_piola":
        # Contravariant Piola map, row-vector convention (psi_phys^T =
        # (1/detJ) * psi_ref^T @ J^T): values genuinely vary per cell,
        # unlike an "identity"-mapped element's.
        values = (reference_values @ jacobians.mT) / jacobian_determinants
        reference_divergences = element.tabulate_basis_divergences(quadrature.points)
        divergences = reference_divergences / jacobian_determinants
        signs = space.local_dof_signs[:, None, :, None]  # (K, 1, N_phi, 1)
        values = values * signs
        divergences = divergences * signs
        gradients = None
    else:
        raise ValueError(f"Unsupported element.mapping {element.mapping!r}.")

    return CellBasis(
        space=space,
        quadrature=quadrature,
        physical_points=physical_points,
        jacobians=jacobians,
        inverse_jacobians=inverse_jacobians,
        jacobian_determinants=jacobian_determinants,
        physical_weights=physical_weights,
        values=values,
        gradients=gradients,
        divergences=divergences,
    )
