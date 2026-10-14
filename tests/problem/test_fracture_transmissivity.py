"""Tests for per-fracture transmissivity tensors."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.mesh import (
    create_dfn_mesh_from_arrays,
    create_dfn_two_fractures_mesh,
    create_fracture_frames,
)
from jax_fem.problem import evaluate_fracture_transmissivity

pytestmark = pytest.mark.unit


def _tilted_network():
    rng = np.random.default_rng(3)
    triangle = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.3, 0.8, 0.0)])
    vertices = []
    for _ in range(3):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        vertices.append(triangle @ rotation.T + rng.normal(size=3))
    cells = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (6, 8, 7)]
    return create_dfn_mesh_from_arrays(np.concatenate(vertices), cells, [5, 1, 9, 9])


def test_isotropic_transmissivity_is_tangential() -> None:
    mesh = _tilted_network()
    transmissivities = jnp.array([2.0, 3.0, 5.0])
    tensors = evaluate_fracture_transmissivity(mesh, transmissivities)
    normals = jnp.expand_dims(mesh.fracture_normals[mesh.cells_to_fractures], axis=-3)

    assert tensors.shape == (4, 1, 3, 3)
    assert np.allclose(tensors, tensors.mT)
    assert np.allclose(normals @ tensors, 0.0)
    assert np.allclose(
        np.trace(tensors[:, 0], axis1=-2, axis2=-1),
        2.0 * transmissivities[mesh.cells_to_fractures],
    )


def test_cells_take_their_fracture_value_in_tag_order() -> None:
    mesh = create_dfn_two_fractures_mesh(2)
    tensors = evaluate_fracture_transmissivity(mesh, jnp.array([1.0, 10.0]))
    first = np.asarray(mesh.cell_tags) == 1

    assert np.allclose(tensors[first, 0], np.diag([1.0, 1.0, 0.0]))
    assert np.allclose(tensors[~first, 0], np.diag([0.0, 10.0, 10.0]))


def test_tensor_transmissivity_in_frames() -> None:
    mesh = _tilted_network()
    frames = create_fracture_frames(mesh)
    local = jnp.array(
        [[[2.0, 0.5], [0.5, 1.0]], [[3.0, 0.0], [0.0, 3.0]], [[1.0, 0.0], [0.0, 4.0]]]
    )
    tensors = evaluate_fracture_transmissivity(mesh, local, frames)
    cell_frames = frames[mesh.cells_to_fractures]
    normals = jnp.expand_dims(mesh.fracture_normals[mesh.cells_to_fractures], axis=-3)

    assert np.allclose(
        cell_frames.mT @ tensors[:, 0] @ cell_frames, local[mesh.cells_to_fractures]
    )
    assert np.allclose(normals @ tensors, 0.0)


def test_isotropic_tensor_input_matches_scalar_input() -> None:
    mesh = _tilted_network()
    scalars = jnp.array([2.0, 3.0, 5.0])
    from_tensors = evaluate_fracture_transmissivity(
        mesh,
        jnp.expand_dims(scalars, axis=(-2, -1)) * jnp.eye(2),
        create_fracture_frames(mesh),
    )
    assert np.allclose(from_tensors, evaluate_fracture_transmissivity(mesh, scalars))


def test_tensor_input_requires_frames() -> None:
    mesh = _tilted_network()
    with pytest.raises(ValueError, match="frames is required"):
        evaluate_fracture_transmissivity(mesh, jnp.ones((3, 2, 2)))


def test_rejects_wrong_number_of_fractures() -> None:
    mesh = _tilted_network()
    with pytest.raises(ValueError, match="transmissivities must have shape"):
        evaluate_fracture_transmissivity(mesh, jnp.ones(2))
