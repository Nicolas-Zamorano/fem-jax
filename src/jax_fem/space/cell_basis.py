"""
Geometry and basis evaluation of a space at one quadrature rule.
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
    Geometry and basis of a finite element space at the quadrature points.

    Attributes
    ----------
    space : FiniteElementSpace
        The finite element space being evaluated.
    quadrature : QuadratureRule
        The quadrature rule the space is evaluated at.
    physical_points : jax.Array
        Quadrature points in physical coordinates, shape ``(K, Q, 1, g)``.
    jacobians : jax.Array
        Jacobians of the affine cell maps, shape ``(K, 1, g, t)``.
    inverse_jacobians : jax.Array
        Inverse Jacobians, or pseudo-inverses when ``g > t``, shape
        ``(K, 1, t, g)``.
    jacobian_determinants : jax.Array
        Jacobian determinants, or area scales ``sqrt(det(J^T J))`` when
        ``g > t``, shape ``(K, 1, 1, 1)``.
    physical_weights : jax.Array
        Physical quadrature weights, shape ``(K, Q, 1, 1)``.
    values : jax.Array
        Basis values, shape ``(1, Q, N_test, value_dim)`` for an
        ``"identity"`` element and ``(K, Q, N_test, value_dim)`` for a
        ``"contravariant_piola"`` element.
    gradients : jax.Array | None
        Physical basis gradients, shape ``(K, 1, N_test, g)`` when constant
        in the point (P0, P1) or ``(K, Q, N_test, g)`` otherwise (P2);
        ``None`` for a ``"contravariant_piola"`` element.
    divergences : jax.Array | None
        Physical basis divergences, shape ``(K, 1, N_test, 1)`` or
        ``(K, Q, N_test, 1)``; ``None`` for an ``"identity"`` element.

    Notes
    -----
    ``g`` is the geometric dimension of the mesh and ``t`` the dimension of
    the reference cell. They are equal for a planar mesh; for cells embedded
    in a higher-dimensional space (``create_embedded_cell_basis``) the
    gradients are tangential and the determinants are unsigned.

    Quantities that are constant along the cell or quadrature axis (true for
    affine, straight-sided triangles) keep that axis with size one and
    broadcast where used.

    For ``"identity"`` elements (P0, P1, P2) the values are the reference
    values, independent of the cell, and the gradients are mapped
    covariantly. For ``"contravariant_piola"`` elements (RT0) the values
    depend on the cell's Jacobian, the divergences replace the gradients,
    and both already include ``space.local_dof_signs``.
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
        """
        Validate array shapes against the space, quadrature, and mapping.

        Raises
        ------
        ValueError
            If a shape is inconsistent or the element mapping is unsupported.
        """
        number_of_cells = self.space.cells_to_dofs.shape[0]
        number_of_points = self.quadrature.points.shape[1]
        number_of_local_dofs = self.space.element.number_of_local_dofs
        geometric_dimension = self.space.mesh.geometric_dimension
        topological_dimension = self.space.element.reference_cell.dimension
        value_dim = self.space.element.value_shape[0]
        mapping = self.space.element.mapping

        check_shape(
            self.physical_points,
            (number_of_cells, number_of_points, 1, geometric_dimension),
            "physical_points",
        )
        check_shape(
            self.jacobians,
            (number_of_cells, 1, geometric_dimension, topological_dimension),
            "jacobians",
        )
        check_shape(
            self.inverse_jacobians,
            (number_of_cells, 1, topological_dimension, geometric_dimension),
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
                (number_of_cells, 1, number_of_local_dofs, geometric_dimension),
                (
                    number_of_cells,
                    number_of_points,
                    number_of_local_dofs,
                    geometric_dimension,
                ),
            )
            if self.divergences is not None:
                raise ValueError(
                    "divergences must be None for an 'identity'-mapped element."
                )
        else:
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
    """
    Require an array with either a constant or a point-varying shape.

    Parameters
    ----------
    array : jax.Array | None
        Array to check; ``None`` is rejected.
    name : str
        Name used in the error message.
    constant_shape : tuple[int, ...]
        Accepted shape when the values are constant in the point.
    varying_shape : tuple[int, ...]
        Accepted shape when the values vary with the point.

    Raises
    ------
    ValueError
        If ``array`` is ``None`` or has neither shape.
    """
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
    Evaluate a finite element space at one quadrature rule.

    Parameters
    ----------
    space : FiniteElementSpace
        The finite element space to evaluate.
    quadrature : QuadratureRule
        Quadrature rule on a reference cell compatible with
        ``space.element.reference_cell``.

    Returns
    -------
    cell_basis : CellBasis
        The geometry and basis at the quadrature points.

    Notes
    -----
    The affine cell map comes from the mesh corner vertices
    (``mesh.cells_to_vertices``), not from ``space.cells_to_dofs``; the two
    coincide only for P1.

    With the row-vector convention, physical points are
    ``origin + xhat @ J^T`` and ``"identity"`` gradients are
    ``grad_ref @ J^-1``. ``"contravariant_piola"`` values are
    ``(1 / det J) ref_value @ J^T`` and divergences ``(1 / det J) ref_div``,
    both multiplied by ``space.local_dof_signs``.
    """
    mesh = space.mesh
    element = space.element
    reference_cell = quadrature.reference_cell

    cell_vertex_coordinates = mesh.vertex_coordinates[mesh.cells_to_vertices]
    origins, second_vertices, third_vertices = jnp.split(
        cell_vertex_coordinates, 3, axis=-3
    )
    # Columns of J are the images of the reference edges from the origin.
    jacobians = jnp.concatenate(
        (second_vertices - origins, third_vertices - origins), axis=-2
    ).mT
    inverse_jacobians = jnp.linalg.inv(jacobians)
    jacobian_determinants = jnp.expand_dims(
        jnp.linalg.det(jacobians), axis=(-2, -1)
    )

    physical_weights = (
        quadrature.weights * reference_cell.measure * jacobian_determinants
    )
    physical_points = origins + quadrature.points @ jacobians.mT

    reference_values = element.tabulate_basis_values(quadrature.points)

    if element.mapping == "identity":
        values = reference_values
        reference_gradients = element.tabulate_basis_gradients(quadrature.points)
        gradients = reference_gradients @ inverse_jacobians
        divergences = None
    elif element.mapping == "contravariant_piola":
        signs = jnp.expand_dims(space.local_dof_signs, axis=(-3, -1))
        values = signs * (reference_values @ jacobians.mT) / jacobian_determinants
        reference_divergences = element.tabulate_basis_divergences(quadrature.points)
        divergences = signs * reference_divergences / jacobian_determinants
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


def compute_tangent_projectors(basis: CellBasis) -> jax.Array:
    """
    Orthogonal projector onto each cell's tangent space.

    Parameters
    ----------
    basis : CellBasis
        Evaluated basis.

    Returns
    -------
    projectors : jax.Array
        ``P = J J^+``, shape ``(K, 1, g, g)``.

    Notes
    -----
    For a planar mesh ``J^+ = J^-1`` and ``P`` is the identity up to
    round-off. For cells embedded in a higher-dimensional space, ``P``
    removes the component normal to the cell, so ``grad(u) @ P`` is the
    tangential gradient of a function ``u`` defined in the ambient space.
    """
    return basis.jacobians @ basis.inverse_jacobians
