"""Tests for cellwise integration, global assembly, and gathering."""

import jax.numpy as jnp
import pytest
from jax.experimental import sparse as jax_sparse

from jax_fem.assembly import (
    assemble_matrix,
    assemble_vector,
    combine_matrix_blocks,
    gather_vector,
    integrate_cellwise,
)
from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import interpolate_function
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


def _make_basis(element_type=LagrangeTriangleP1, exactness_degree: int = 2):
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = element_type(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(exactness_degree)
    return create_cell_basis(space, quadrature)


def _make_problem(diffusion=None, source: float = 0.0) -> EllipticProblem:
    return EllipticProblem(
        diffusion=constant_tensor_coefficient(
            jnp.eye(2) if diffusion is None else diffusion
        ),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(source),
        dirichlet_conditions=(),
    )


def _local_matrices(test_basis, trial_basis, problem):
    coefficients = evaluate_elliptic_coefficients(problem, test_basis.physical_points)
    return integrate_cellwise(
        elliptic_bilinear_form(test_basis, trial_basis, coefficients),
        test_basis.physical_weights,
    )


def _local_vectors(basis, problem):
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    return integrate_cellwise(
        elliptic_linear_form(basis, coefficients), basis.physical_weights
    )


@pytest.mark.unit
def test_integrate_cellwise_keeps_the_quadrature_axis() -> None:
    pointwise_values = jnp.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    physical_weights = jnp.array([[[[0.25]], [[0.75]]]])
    result = integrate_cellwise(pointwise_values, physical_weights)
    expected = jnp.array(
        [[[[0.25 * 1.0 + 0.75 * 3.0], [0.25 * 2.0 + 0.75 * 4.0]]]]
    )
    assert result.shape == (1, 1, 2, 1)
    assert jnp.allclose(result, expected, atol=1e-14, rtol=1e-12)


@pytest.mark.unit
def test_integrate_cellwise_accepts_leading_batch_dimensions() -> None:
    pointwise_values = jnp.arange(24.0).reshape(2, 1, 3, 2, 2)
    physical_weights = jnp.array([[[[0.5]], [[0.25]], [[0.25]]]])
    result = integrate_cellwise(pointwise_values, physical_weights)
    assert result.shape == (2, 1, 1, 2, 2)
    expected = integrate_cellwise(pointwise_values[1], physical_weights)
    assert jnp.allclose(result[1], expected)


@pytest.mark.unit
def test_assemble_matrix_matches_hand_derived_global_stiffness() -> None:
    basis = _make_basis()
    matrix = assemble_matrix(
        _local_matrices(basis, basis, _make_problem()), basis.space, basis.space
    )

    assert isinstance(matrix, jax_sparse.BCOO)
    assert matrix.shape == (4, 4)
    expected = jnp.array(
        [
            [1.0, -0.5, 0.0, -0.5],
            [-0.5, 1.0, -0.5, 0.0],
            [0.0, -0.5, 1.0, -0.5],
            [-0.5, 0.0, -0.5, 1.0],
        ]
    )
    assert jnp.allclose(matrix.todense(), expected, atol=1e-14, rtol=1e-12)


@pytest.mark.unit
def test_assemble_vector_matches_hand_derived_global_load_vector() -> None:
    basis = _make_basis()
    vector = assemble_vector(
        _local_vectors(basis, _make_problem(jnp.zeros((2, 2)), source=1.0)),
        basis.space,
    )

    assert vector.shape == (4, 1)
    expected = jnp.array([[1.0 / 3.0], [1.0 / 6.0], [1.0 / 3.0], [1.0 / 6.0]])
    assert jnp.allclose(vector, expected, atol=1e-14, rtol=1e-12)


@pytest.mark.unit
def test_assemble_matrix_matches_independent_dense_scatter() -> None:
    basis = _make_basis()
    local_matrices = _local_matrices(basis, basis, _make_problem())
    matrix = assemble_matrix(local_matrices, basis.space, basis.space)

    cells_to_dofs = basis.space.cells_to_dofs
    number_of_dofs = basis.space.number_of_dofs
    dense_reference = jnp.zeros((number_of_dofs, number_of_dofs))
    for cell_index in range(cells_to_dofs.shape[0]):
        for local_i in range(3):
            for local_j in range(3):
                global_i = int(cells_to_dofs[cell_index, local_i])
                global_j = int(cells_to_dofs[cell_index, local_j])
                dense_reference = dense_reference.at[global_i, global_j].add(
                    local_matrices[cell_index, 0, local_i, local_j]
                )

    assert jnp.allclose(matrix.todense(), dense_reference, atol=1e-14, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.property
def test_assembled_matrix_is_symmetric_for_symmetric_diffusion() -> None:
    basis = _make_basis()
    problem = _make_problem(jnp.array([[2.0, 0.3], [0.3, 1.0]]))
    dense = assemble_matrix(
        _local_matrices(basis, basis, problem), basis.space, basis.space
    ).todense()
    assert jnp.allclose(dense, dense.T, atol=1e-14, rtol=1e-12)
    assert jnp.allclose(jnp.sum(dense, axis=-1), 0.0, atol=1e-14)


@pytest.mark.unit
@pytest.mark.property
def test_gather_vector_is_the_adjoint_of_assemble_vector() -> None:
    basis = _make_basis(LagrangeTriangleP2)
    local_vectors = _local_vectors(basis, _make_problem(source=1.3))
    global_vector = jnp.linspace(-1.0, 2.0, basis.space.number_of_dofs)[:, None]

    assembled = assemble_vector(local_vectors, basis.space)
    gathered = gather_vector(global_vector, basis.space)

    assert gathered.shape == local_vectors.shape
    assert jnp.sum(assembled * global_vector) == pytest.approx(
        float(jnp.sum(local_vectors * gathered)), rel=1e-13
    )


@pytest.mark.unit
def test_matrix_free_action_matches_assembled_matrix() -> None:
    basis = _make_basis(LagrangeTriangleP2)
    local_matrices = _local_matrices(basis, basis, _make_problem())
    matrix = assemble_matrix(local_matrices, basis.space, basis.space)
    x = jnp.cos(jnp.arange(basis.space.number_of_dofs, dtype=float))[:, None]

    local_action = local_matrices @ gather_vector(x, basis.space)
    action = assemble_vector(local_action, basis.space)
    assert jnp.allclose(action, matrix @ x, atol=1e-13)


@pytest.mark.unit
def test_assemble_vector_and_gather_vector_accept_leading_batch_dimensions() -> None:
    basis = _make_basis()
    local_vectors = _local_vectors(basis, _make_problem(source=1.0))
    batched = jnp.stack((local_vectors, 2.0 * local_vectors))

    assembled = assemble_vector(batched, basis.space)
    assert assembled.shape == (2, basis.space.number_of_dofs, 1)
    assert jnp.allclose(assembled[1], 2.0 * assemble_vector(local_vectors, basis.space))

    gathered = gather_vector(assembled, basis.space)
    assert gathered.shape == (2,) + local_vectors.shape


@pytest.mark.unit
def test_assemble_matrix_rejects_local_matrices_without_quadrature_axis() -> None:
    basis = _make_basis()
    local_matrices = _local_matrices(basis, basis, _make_problem())
    with pytest.raises(ValueError, match="local_matrices must have shape"):
        assemble_matrix(jnp.squeeze(local_matrices, axis=-3), basis.space, basis.space)


@pytest.mark.unit
def test_assemble_vector_rejects_wrong_local_shape() -> None:
    basis = _make_basis()
    local_vectors = _local_vectors(basis, _make_problem(source=1.0))
    with pytest.raises(ValueError, match="local_vectors must have shape"):
        assemble_vector(jnp.squeeze(local_vectors, axis=-3), basis.space)


@pytest.mark.unit
def test_petrov_galerkin_matrix_acts_like_galerkin_on_a_linear_function() -> None:
    """A P1-test x P2-trial matrix applied to the P2 interpolant of a linear
    function equals the P1 stiffness applied to its P1 interpolant."""
    test_basis = _make_basis(LagrangeTriangleP1, exactness_degree=2)
    trial_basis = create_cell_basis(
        create_finite_element_space(
            test_basis.space.mesh,
            LagrangeTriangleP2(reference_cell=ReferenceTriangle()),
        ),
        test_basis.quadrature,
    )
    problem = _make_problem()

    def linear(points):
        return 1.0 + 2.0 * points[..., [0]] - 3.0 * points[..., [1]]

    petrov_galerkin = assemble_matrix(
        _local_matrices(test_basis, trial_basis, problem),
        test_basis.space,
        trial_basis.space,
    )
    galerkin = assemble_matrix(
        _local_matrices(test_basis, test_basis, problem),
        test_basis.space,
        test_basis.space,
    )

    assert petrov_galerkin.shape == (
        test_basis.space.number_of_dofs,
        trial_basis.space.number_of_dofs,
    )
    p2_values = interpolate_function(linear, trial_basis.space).dof_values
    p1_values = interpolate_function(linear, test_basis.space).dof_values
    assert jnp.allclose(petrov_galerkin @ p2_values, galerkin @ p1_values, atol=1e-13)


@pytest.mark.unit
def test_combine_matrix_blocks_matches_dense_block_layout() -> None:
    a = jax_sparse.BCOO.fromdense(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    b = jax_sparse.BCOO.fromdense(jnp.array([[5.0], [6.0]]))
    c = jax_sparse.BCOO.fromdense(jnp.array([[7.0, 8.0]]))

    combined = combine_matrix_blocks(((a, b), (c, None)))

    expected = jnp.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [7.0, 8.0, 0.0]])
    assert combined.shape == (3, 3)
    assert jnp.allclose(combined.todense(), expected)


@pytest.mark.unit
def test_combine_matrix_blocks_keeps_duplicate_entries_summed() -> None:
    basis = _make_basis()
    matrix = assemble_matrix(
        _local_matrices(basis, basis, _make_problem()), basis.space, basis.space
    )
    combined = combine_matrix_blocks(((matrix, None), (None, matrix)))
    dense = matrix.todense()
    assert jnp.allclose(combined.todense()[:4, :4], dense)
    assert jnp.allclose(combined.todense()[4:, 4:], dense)
    assert jnp.allclose(combined.todense()[:4, 4:], 0.0)


@pytest.mark.unit
def test_combine_matrix_blocks_rejects_inconsistent_sizes() -> None:
    a = jax_sparse.BCOO.fromdense(jnp.ones((2, 2)))
    b = jax_sparse.BCOO.fromdense(jnp.ones((3, 1)))
    with pytest.raises(ValueError, match="Block \\(0, 1\\) has shape"):
        combine_matrix_blocks(((a, b),))


@pytest.mark.unit
def test_combine_matrix_blocks_rejects_an_all_none_block_row() -> None:
    a = jax_sparse.BCOO.fromdense(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="block row 1 contains only None"):
        combine_matrix_blocks(((a,), (None,)))
