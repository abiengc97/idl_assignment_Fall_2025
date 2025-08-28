import numpy as np
import torch

from mytorch import autograd_engine
import mytorch.nn as nn
from helpers import *
from mytorch.functional_hw1 import *


# Activation Layer test - for Sigmoid, Tanh and ReLU

def test_identity_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet
    l1_out = l1(x)
    test_act = nn.Identity(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Identity()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)

    return True


def test_identity_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Identity(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Identity()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)

    return True


def test_sigmoid_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Sigmoid(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Sigmoid()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)

    return True


def test_sigmoid_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Sigmoid(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Sigmoid()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)

    return True


def test_sigmoid_negative_zero_positive():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float64)

    test_act = nn.Sigmoid(autograd)
    sig_out = test_act(x)
    autograd.backward(1.0)

    # Torch input
    torch_x = torch.DoubleTensor(x)
    torch_x.requires_grad_(True)
    torch_sig_out = torch.sigmoid(torch_x)
    torch_sig_out.sum().backward()

    compare_np_torch(sig_out, torch_sig_out)

    my_grad = autograd.gradient_buffer.get_param(x)
    torch_grad = torch_x.grad
    compare_np_torch(my_grad, torch_grad)

    return True


def test_tanh_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Tanh(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Tanh()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)
    return True


def test_tanh_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Tanh(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Tanh()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)

    return True


def test_tanh_negative_zero_positive():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float64)

    test_act = nn.Tanh(autograd)
    tanh_out = test_act(x)
    autograd.backward(1.0)  # Pass scalar 1.0 to accumulate gradient in your engine

    # Torch input
    torch_x = torch.DoubleTensor(x)
    torch_x.requires_grad_(True)
    torch_tanh_out = torch.tanh(torch_x)
    torch_tanh_out.sum().backward()

    # Compare forward outputs
    compare_np_torch(tanh_out, torch_tanh_out)

    # Compare backward (gradient on x)
    my_grad = autograd.gradient_buffer.get_param(x)
    torch_grad = torch_x.grad
    compare_np_torch(my_grad, torch_grad)

    return True


def test_relu_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.ReLU(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    # print(torch_l1_out.shape)
    torch_act = torch.nn.ReLU()
    torch_a1_out = torch_act(torch_l1_out)
    # print(torch_a1_out.shape)

    compare_np_torch(a1_out, torch_a1_out)

    return True


def test_relu_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.ReLU(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.ReLU()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)

    return True


def test_relu_negative_zero_positive():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float64)

    test_act = nn.ReLU(autograd)
    relu_out = test_act(x)
    autograd.backward(1.0)

    # Torch input
    torch_x = torch.DoubleTensor(x)
    torch_x.requires_grad_(True)
    torch_relu_out = torch.nn.functional.relu(torch_x)
    torch_relu_out.sum().backward()

    compare_np_torch(relu_out, torch_relu_out)

    my_grad = autograd.gradient_buffer.get_param(x)
    torch_grad = torch_x.grad
    compare_np_torch(my_grad, torch_grad)

    return True
