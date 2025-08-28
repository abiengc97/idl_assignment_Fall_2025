import numpy as np
import torch
from mytorch.autograd_engine import Autograd, Operation
import mytorch.functional_hw1 as F
from helpers import compare_np_torch


def test_add_operation():
    autograd_engine = Autograd()
    x = np.random.randn(1, 5)
    y = np.random.randn(1, 5)

    z = x + y

    autograd_engine.add_operation(
        inputs=[x, y],
        output=z,
        gradients_to_update=[None, None],
        backward_operation=F.add_backward
    )

    assert len(autograd_engine.gradient_buffer.memory) == 2

    assert len(autograd_engine.operation_list) == 1

    operation = autograd_engine.operation_list[0]

    assert type(operation) == Operation

    assert len(operation.inputs) == 2
    assert np.array_equal(operation.inputs[0], x) and np.array_equal(operation.inputs[1], y)
    assert np.array_equal(operation.output, z)
    assert len(operation.gradients_to_update) == 2
    assert operation.backward_operation == F.add_backward

    return True


def test_backward():
    """
    Basic Test: Original Tests
    """
    autograd_engine = Autograd()
    x1 = np.random.randn(1, 5)
    y1 = np.random.randn(1, 5)

    z1 = x1 + y1

    autograd_engine.add_operation(
        inputs=[x1, y1],
        output=z1,
        gradients_to_update=[None, None],
        backward_operation=F.add_backward
    )

    assert len(autograd_engine.operation_list) == 1
    assert len(autograd_engine.gradient_buffer.memory) == 2

    autograd_engine.backward(1)
    dy1 = autograd_engine.gradient_buffer.get_param(y1)

    torch_x1 = torch.DoubleTensor(torch.tensor(x1, requires_grad=True))
    torch_y1 = torch.DoubleTensor(torch.tensor(y1, requires_grad=True))
    torch_x1.retain_grad()
    torch_y1.retain_grad()

    torch_z1 = torch_x1 + torch_y1
    torch_z1.sum().backward()

    compare_np_torch(z1, torch_z1)
    compare_np_torch(dy1, torch_y1.grad)
    compare_np_torch(autograd_engine.gradient_buffer.get_param(x1), torch_x1.grad)

    return True


def test_backward_gradient_update():
    autograd_engine = Autograd()
    divergence = 1
    x1 = np.array([
        [-1.],
        [2.],
        [1.]], dtype="f")
    y1 = np.array([
        [-4.],
        [1.],
        [3.]], dtype="f")
    dy1 = np.array([
        [-1.],
        [2.],
        [3.]], dtype="f")

    z1 = x1 + y1

    autograd_engine.add_operation(
        inputs=[x1, y1],
        output=z1,
        gradients_to_update=[None, dy1],
        backward_operation=F.add_backward
    )

    autograd_engine.backward(divergence)

    # Testing Updates on Updatable gradients_to_update
    dy1_updated_gradient = np.array([
        [0.],
        [3.],
        [4.]], dtype='f') + (divergence - 1)
    dy1_updated = autograd_engine.operation_list[0].gradients_to_update[1]

    assert dy1_updated_gradient.shape == dy1_updated.shape, f'numpy: {dy1_updated_gradient.shape}, numpy: {dy1_updated.shape}'
    assert np.allclose(dy1_updated_gradient, dy1_updated, rtol=1e-05, atol=1e-10), "NOT ALL CLOSE:\n{}\n{}".format(dy1_updated_gradient, dy1_updated)
    assert np.abs(dy1_updated_gradient - dy1_updated).sum() < 1e-10, "{} vs {}, diff: {}".format(dy1_updated_gradient, dy1_updated, np.abs(dy1_updated_gradient - dy1_updated).sum())

    # Testing Updates on Un-updatable gradients_to_update
    dx1_updated_gradient = np.array([
        [1.],
        [1.],
        [1.]], dtype='f') * divergence

    dx1_updated = autograd_engine.gradient_buffer.get_param(x1)
    assert dx1_updated_gradient.shape == dx1_updated.shape, f'numpy: {dx1_updated_gradient.shape}, numpy: {dx1_updated.shape}'
    assert np.allclose(dx1_updated_gradient, dx1_updated, rtol=1e-05, atol=1e-10), "NOT ALL CLOSE:\n{}\n{}".format(dx1_updated_gradient, dx1_updated)
    assert np.abs(dx1_updated_gradient - dx1_updated).sum() < 1e-10, "{} vs {}, diff: {}".format(dx1_updated_gradient, dx1_updated, np.abs(dx1_updated_gradient - dx1_updated).sum())

    return True
