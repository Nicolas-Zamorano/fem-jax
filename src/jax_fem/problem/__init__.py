"""PDE coefficient and boundary-condition data."""

from jax_fem.problem.boundary_conditions import (
    DirichletCondition,
    create_full_boundary_dirichlet_condition,
)
from jax_fem.problem.dfn import (
    create_dfn_two_fractures_problem,
    evaluate_fracture_transmissivity,
)
from jax_fem.problem.elliptic import (
    EllipticCoefficientValues,
    EllipticProblem,
    ScalarCoefficient,
    TensorCoefficient,
    VectorCoefficient,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    evaluate_elliptic_coefficients,
    isotropic_tensor_coefficient,
    zero_scalar_coefficient,
)
from jax_fem.problem.manufactured import (
    create_poisson_cos_sin_problem,
    create_poisson_exponential_problem,
    create_poisson_l_shaped_singular_problem,
    create_poisson_sin_sin_problem,
    create_poisson_singular_problem,
    create_simple_elliptic_problem,
)

__all__ = [
    "EllipticProblem",
    "EllipticCoefficientValues",
    "ScalarCoefficient",
    "VectorCoefficient",
    "TensorCoefficient",
    "evaluate_elliptic_coefficients",
    "constant_scalar_coefficient",
    "constant_vector_coefficient",
    "constant_tensor_coefficient",
    "isotropic_tensor_coefficient",
    "zero_scalar_coefficient",
    "DirichletCondition",
    "create_full_boundary_dirichlet_condition",
    "create_poisson_sin_sin_problem",
    "create_poisson_cos_sin_problem",
    "create_poisson_exponential_problem",
    "create_poisson_singular_problem",
    "create_poisson_l_shaped_singular_problem",
    "create_simple_elliptic_problem",
    "evaluate_fracture_transmissivity",
    "create_dfn_two_fractures_problem",
]
