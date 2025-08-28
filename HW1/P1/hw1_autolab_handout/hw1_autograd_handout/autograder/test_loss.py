import numpy as np
import torch

from mytorch import autograd_engine
import mytorch.nn as nn
from helpers import *
from mytorch.functional_hw1 import *


def test_softmaxXentropy_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    y = np.array([[0., 0., 1., 0., 0.]])
    l1_out = l1(x)

    test_loss = nn.SoftmaxCrossEntropy(autograd)
    a1_out = test_loss(y, l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_y = torch.LongTensor(np.array([2]))
    torch_l1_out = torch_l1(torch_x)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_a1_out = torch_loss(torch_l1_out, torch_y).reshape(1, )
    torch_a1_out.backward()

    compare_np_torch(a1_out, torch_a1_out)

    return True


def test_softmaxXentropy_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    y = np.array([[0., 0., 1., 0., 0.]])
    l1_out = l1(x)

    # Torch input
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_y = torch.LongTensor(np.array([2]))
    torch_l1_out = torch_l1(torch_x)

    test_loss = nn.SoftmaxCrossEntropy(autograd)
    a1_out = test_loss(y, l1_out)
    test_loss.backward()

    torch_loss = torch.nn.CrossEntropyLoss()

    torch_a1_out = torch_loss(torch_l1_out, torch_y).reshape(1, )
    torch_a1_out.backward()

    compare_np_torch(a1_out, torch_a1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)

    return True


def test_softmaxXentropy_forward_batch():
    # Test input with multiple samples
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((3, 5))  # Batch of 3 samples
    y = np.array([[0., 0., 1., 0., 0.], [0., 1., 0., 0., 0.], [1., 0., 0., 0., 0.]])  # One-hot labels
    l1_out = l1(x)

    test_loss = nn.SoftmaxCrossEntropy(autograd)
    a1_out = test_loss(y, l1_out)

    # Torch equivalent
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_y = torch.LongTensor([2, 1, 0])  # Class indices
    torch_l1_out = torch_l1(torch_x)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_a1_out = torch_loss(torch_l1_out, torch_y).reshape(1, )
    torch_a1_out.backward()

    compare_np_torch(a1_out, torch_a1_out)
    return True


def test_softmaxXentropy_forward_edge_cases():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.array([[1e10, -1e10, 0, 1e5, -1e5]])  # Extreme values
    y = np.array([[0., 0., 1., 0., 0.]])  # One-hot label
    l1_out = l1(x)

    test_loss = nn.SoftmaxCrossEntropy(autograd)
    a1_out = test_loss(y, l1_out)

    # Torch equivalent
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_y = torch.LongTensor([2])  # Class index
    torch_l1_out = torch_l1(torch_x)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_a1_out = torch_loss(torch_l1_out, torch_y).reshape(1, )
    torch_a1_out.backward()

    compare_np_torch(a1_out, torch_a1_out)
    return True


def test_softmaxXentropy_backward_gradient_magnitude():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.array([[100, 200, -300, 400, -500]])  # Large values
    y = np.array([[0., 0., 1., 0., 0.]])  # One-hot label
    l1_out = l1(x)

    test_loss = nn.SoftmaxCrossEntropy(autograd)
    a1_out = test_loss(y, l1_out)
    test_loss.backward()

    # Torch equivalent
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_y = torch.LongTensor([2])  # Class index
    torch_l1_out = torch_l1(torch_x)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_a1_out = torch_loss(torch_l1_out, torch_y).reshape(1, )
    torch_a1_out.backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    return True


def test_softmaxXentropy_forward_batch_size_one():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))  # Batch size of 1
    y = np.array([[0., 1., 0., 0., 0.]])  # One-hot label
    l1_out = l1(x)

    test_loss = nn.SoftmaxCrossEntropy(autograd)
    a1_out = test_loss(y, l1_out)

    # Torch equivalent
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_y = torch.LongTensor([1])  # Class index
    torch_l1_out = torch_l1(torch_x)

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_a1_out = torch_loss(torch_l1_out, torch_y).reshape(1, )
    torch_a1_out.backward()

    compare_np_torch(a1_out, torch_a1_out)
    return True


def test_softmaxXentropy_incorrect_ground_truth():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))  # Single sample
    y = np.array([2])  # Incorrect dimension, should be one-hot encoded

    test_loss = nn.SoftmaxCrossEntropy(autograd)

    try:
        a1_out = test_loss(y, l1(x))
    except ValueError as e:
        print("Caught expected ValueError:", e)
        return True

    assert False, "Expected ValueError for incorrect ground truth dimensions"
