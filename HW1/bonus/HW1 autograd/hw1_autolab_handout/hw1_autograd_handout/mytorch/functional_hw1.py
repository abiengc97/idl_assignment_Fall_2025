import numpy as np
from mytorch.autograd_engine import Autograd

"""
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with your other code.
    
    Optional:
    - You can account for broadcasting, but it is not required in the first bonus.
"""
def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def identity_backward(grad_output, a):
    """Backward for identity. Already implemented."""

    return grad_output


def add_backward(grad_output, a, b):
    """Backward for addition. Already implemented."""

    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * np.ones(b.shape)

    return a_grad, b_grad


def sub_backward(grad_output, a, b):
    """Backward for subtraction"""

    return NotImplementedError


def matmul_backward(grad_output, a, b):
    """Backward for matrix multiplication"""

    return NotImplementedError

def transpose_backward(grad_output, a):
    """Backward for matrix transpose"""
    return NotImplementedError

def mul_backward(grad_output, a, b):
    """Backward for multiplication"""

    return NotImplementedError


def div_backward(grad_output, a, b):
    """Backward for division"""

    return NotImplementedError


def log_backward(grad_output, a):
    """Backward for log"""

    return NotImplementedError


def exp_backward(grad_output, a):
    """Backward of exponential"""

    return NotImplementedError


def max_backward(grad_output, a):
    """Backward of max"""

    return NotImplementedError


def sum_backward(grad_output, a):
    """Backward of sum"""

    return NotImplementedError

def tanh_backward(grad_output, a):
    """Backward of tanh"""
    return NotImplementedError

def SoftmaxCrossEntropy_backward(grad_output, pred, ground_truth):
    """
    TODO: implement Softmax CrossEntropy Loss here. You may want to modify the function signature to include more inputs.
    Note: Since the gradient of the Softmax CrossEntropy Loss is is straightforward to compute, you may choose to implement this directly rather than rely on the backward functions of more primitive operations.
    Note: Do we need the grad_output for the SoftmaxCross calculations in this code snippet?
    """
    return NotImplementedError

# Functions foor visualization (Do Not Modify)
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=float)  # store as NumPy array
        self.grad = np.zeros_like(self.data)     # gradient same shape
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.label = label
    
    def backward(self):
    
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
            
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            a_grad, b_grad = add_backward(out.grad, self.data, other.data)
            self.grad += a_grad
            other.grad += b_grad
        out._backward = _backward

        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            a_grad, b_grad = mul_backward(out.grad, self.data, other.data)
            self.grad += a_grad
            other.grad += b_grad
        out._backward = _backward
        return out
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            a_grad, b_grad = div_backward(out.grad, self.data, other.data)
            self.grad += a_grad
            other.grad += b_grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += max_backward(out.grad, self.data)
        out._backward = _backward

        return out
    def tanh(self):
        out = Value(np.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += tanh_backward(out.grad, self.data)
        out._backward = _backward

        return out
    
    def log(self):
        out = Value(np.log(self.data), (self,), 'log')

        def _backward():
            grad = log_backward(out.grad, self.data)
            self.grad += unbroadcast(grad, self.data.shape)
        out._backward = _backward

        return out
    def exp(self):
        out = Value(np.exp(self.data), (self,), 'exp')

        def _backward():
            grad = exp_backward(out.grad, self.data)
            self.grad += unbroadcast(grad, self.data.shape)
        out._backward = _backward

        return out
    
        # Unary and reflected ops
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): return Value(other) / self

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
