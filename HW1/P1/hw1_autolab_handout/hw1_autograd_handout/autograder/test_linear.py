import numpy as np
import torch

from mytorch import autograd_engine
import mytorch.nn as nn
from helpers import *
from mytorch.functional_hw1 import *


def test_linear_layer_forward():
    np.random.seed(0)
    x = np.random.random((1, 5))

    autograd = autograd_engine.Autograd()
    l1 = nn.Linear(5, 5, autograd)
    l1_out = l1(x)

    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)

    compare_np_torch(l1_out, torch_l1_out)
    return True


def test_linear_layer_backward():
    np.random.seed(0)
    x = np.random.random((1, 5))

    autograd = autograd_engine.Autograd()
    l1 = nn.Linear(5, 5, autograd)
    l1_out = l1(x)
    autograd.backward(1)

    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_l1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    return True


def test_linear_skip_forward():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    autograd.zero_grad()
    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    output = l1_out + x
    autograd.add_operation(inputs=[l1_out, x], output=output, gradients_to_update=[None, None], backward_operation=add_backward)

    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))

    torch_x = torch.DoubleTensor(x)
    torch_x.requires_grad = True

    torch_l1_out = torch_l1(torch_x)
    torch_output = torch_l1_out + torch_x

    compare_np_torch(output, torch_output)

    return True


def test_linear_skip_backward():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    autograd.zero_grad()
    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    output = l1_out + x
    autograd.add_operation(inputs=[l1_out, x], output=output, gradients_to_update=[None, None], backward_operation=add_backward)
    autograd.backward(1)

    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))

    torch_x = torch.DoubleTensor(x)
    torch_x.requires_grad = True

    torch_l1_out = torch_l1(torch_x)
    torch_output = torch_l1_out + torch_x
    torch_output.sum().backward()

    compare_np_torch(l1_out, torch_l1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    compare_np_torch(autograd.gradient_buffer.get_param(x), torch_x.grad)  # skip connections work'''
    return True


def test_linear_layer_forward_different_sizes():
    np.random.seed(0)
    x = np.random.random((2, 10))  # Input with 2 samples and 10 features

    autograd = autograd_engine.Autograd()
    l1 = nn.Linear(10, 7, autograd)  # Linear layer mapping from 10 to 7 dimensions
    l1_out = l1(x)

    torch_l1 = torch.nn.Linear(10, 7)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)

    compare_np_torch(l1_out, torch_l1_out)
    return True


def test_linear_basic_backward():
    np.random.seed(0)
    x = np.random.random((3, 8))  # Input with 3 samples and 8 features

    autograd = autograd_engine.Autograd()
    l1 = nn.Linear(8, 4, autograd)  # Linear layer mapping from 8 to 4 dimensions
    l1_out = l1(x)
    autograd.backward(np.ones_like(l1_out))  # Backward pass with a gradient of ones

    torch_l1 = torch.nn.Linear(8, 4)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_l1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    return True


def test_linear_bias_broadcasting():
    np.random.seed(0)
    x = np.random.random((4, 6))  # Input with 4 samples and 6 features

    autograd = autograd_engine.Autograd()
    l1 = nn.Linear(6, 5, autograd)  # Linear layer mapping from 6 to 5 dimensions
    l1_out = l1(x)

    torch_l1 = torch.nn.Linear(6, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)

    compare_np_torch(l1_out, torch_l1_out)
    return True


def test_linear_edge_cases():
    autograd = autograd_engine.Autograd()

    # Empty input
    x_empty = np.empty((0, 5))
    l1 = nn.Linear(5, 5, autograd)
    l1_out_empty = l1(x_empty)
    assert l1_out_empty.shape == (0, 5), "Empty input test failed"

    # Large values
    x_large = np.random.random((1, 5)) * 1e6
    l1_out_large = l1(x_large)
    assert not np.isnan(l1_out_large).any(), "Large values test failed"

    # Small values
    x_small = np.random.random((1, 5)) * 1e-6
    l1_out_small = l1(x_small)
    assert not np.isnan(l1_out_small).any(), "Small values test failed"

    return True


def test_linear_with_non_linearity():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    # Sequential operations: Linear -> ReLU -> Linear
    l1 = nn.Linear(5, 3, autograd)
    l2 = nn.Linear(3, 2, autograd)
    x = np.random.random((2, 5))

    l1_out = l1(x)
    relu_out = np.maximum(0, l1_out)  # ReLU activation
    autograd.add_operation(inputs=[l1_out], output=relu_out, gradients_to_update=[None], backward_operation=max_backward)
    l2_out = l2(relu_out)

    # Torch equivalent
    torch_l1 = torch.nn.Linear(5, 3)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))

    torch_l2 = torch.nn.Linear(3, 2)
    torch_l2.weight = torch.nn.Parameter(torch.DoubleTensor(l2.W))
    torch_l2.bias = torch.nn.Parameter(torch.DoubleTensor(l2.b.squeeze()))

    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_relu_out = torch.nn.functional.relu(torch_l1_out)
    torch_l2_out = torch_l2(torch_relu_out)

    compare_np_torch(l2_out, torch_l2_out)
    return True
