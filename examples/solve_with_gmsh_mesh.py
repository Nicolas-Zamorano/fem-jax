"""Solve the SimpleEllipticProblem manufactured problem on a Gmsh-built mesh.

Same workflow as ``solve_simple_elliptic_problem.py`` (mesh -> element ->
space -> basis -> problem -> assembly -> Dirichlet condensation ->
user-chosen solve -> reconstruction -> error measurement); only the mesh
source differs. The unit square is built directly with the Gmsh Python API
(Section 7.4) rather than ``create_structured_unit_square_mesh``, with its
four sides tagged as separate physical curves so the boundary can still be
inspected per side if needed, even though ``SimpleEllipticProblem`` applies
the same Dirichlet value on all of them.

Requires the optional 'gmsh' dependency: pip install 'jax-fem[gmsh]'.

Run with:

    uv run python examples/solve_with_gmsh_mesh.py
"""

import jax.numpy as jnp

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import (
    compute_h1_seminorm_error,
    compute_l2_error,
    plot_fem_error,
    plot_fem_error_3d,
    save_plots,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import TriangleMesh, create_triangle_mesh_from_gmsh_model
from jax_fem.problem import (
    create_simple_elliptic_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space


def build_gmsh_unit_square_mesh(mesh_size: float = 0.05) -> TriangleMesh:
    """Build a unit square directly with the Gmsh Python API.

    Four boundary curves ("bottom", "right", "top", "left") and one
    surface ("domain") are tagged as physical groups so
    ``create_triangle_mesh_from_gmsh_model`` (Section 7.4) can carry those
    tags into the resulting ``TriangleMesh``. Gmsh is initialized and
    finalized entirely within this function, independent of any other
    live Gmsh session.

    Parameters
    ----------
    mesh_size:
        Target element size passed to every corner point; smaller values
        give a finer, denser triangulation.

    Returns
    -------
    TriangleMesh
    """
    try:
        import gmsh
    except ImportError as error:
        raise ImportError(
            "This example requires the optional 'gmsh' dependency. "
            "Install it with: pip install 'jax-fem[gmsh]'."
        ) from error

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
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
        return create_triangle_mesh_from_gmsh_model(gmsh.model)
    finally:
        gmsh.finalize()


def main(mesh_size: float = 0.05) -> None:

    mesh = build_gmsh_unit_square_mesh(mesh_size)

    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())

    space = create_finite_element_space(mesh, element)

    quadrature = space.element.reference_cell.create_quadrature(5)
    basis = create_cell_basis(space, quadrature)

    problem = create_simple_elliptic_problem(mesh)

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)

    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )

    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)

    full_dof_values = expand_condensed_solution(free_dof_values, condensed)
    solution = FiniteElementFunction(space=space, dof_values=full_dof_values)

    l2_error = float(compute_l2_error(solution, basis, problem))
    h1_error = float(compute_h1_seminorm_error(solution, basis, problem))

    error_figure = plot_fem_error(solution, basis, problem, mesh)

    figure_3d = plot_fem_error_3d(solution, basis, problem, mesh)

    directory = save_plots(
        {"error_mesh": error_figure, "error_mesh_3d": figure_3d},
        directory="experiments",
        name=f"{problem.name}_gmsh",
    )

    number_of_cells = mesh.cells_to_vertices.shape[0]
    print(f"gmsh target element size: {mesh_size} ({number_of_cells} cells)")
    print(
        f"degrees of freedom: {space.number_of_dofs} total, "
        f"{condensed.free_dofs.shape[0]} free"
    )
    print(f"L2 error:           {l2_error:.6e}")
    print(f"H1 seminorm error:  {h1_error:.6e}")

    print("Plots saved to ", directory)


if __name__ == "__main__":
    main()
