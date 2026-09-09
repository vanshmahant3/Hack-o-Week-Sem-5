import math
from typing import Set, Tuple, List, Callable, Union

class Value:
    """Scalar Autograd Engine Node maintaining a dynamic computational graph (DAG).
    Implements reverse-mode automatic differentiation using the Multivariate Chain Rule.
    """
    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: Union['Value', float, int]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Multivariate chain rule addition: dL/dx += 1 * dL/dout
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other: Union[float, int]) -> 'Value':
        return self + other

    def __mul__(self, other: Union['Value', float, int]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Product rule: dL/dx += other.data * dL/dout
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other: Union[float, int]) -> 'Value':
        return self * other

    def __pow__(self, other: Union[int, float]) -> 'Value':
        assert isinstance(other, (int, float)), "Power only supports int/float exponents."
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Power rule: dL/dx += other * (x ** (other - 1)) * dL/dout
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def __neg__(self) -> 'Value':
        return self * -1.0

    def __sub__(self, other: Union['Value', float, int]) -> 'Value':
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> 'Value':
        return Value(other) + (-self)

    def __truediv__(self, other: Union['Value', float, int]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other: Union[float, int]) -> 'Value':
        return Value(other) * (self ** -1)

    def relu(self) -> 'Value':
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            # ReLU derivative: 1 if x > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self) -> 'Value':
        # Clamped for numerical stability
        clamped_data = max(min(self.data, 50.0), -50.0)
        s = 1.0 / (1.0 + math.exp(-clamped_data))
        out = Value(s, (self,), 'Sigmoid')

        def _backward():
            # Sigmoid derivative: s * (1 - s) * dL/dout
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward

        return out

    def tanh(self) -> 'Value':
        clamped_data = max(min(self.data, 50.0), -50.0)
        t = math.tanh(clamped_data)
        out = Value(t, (self,), 'Tanh')

        def _backward():
            # Tanh derivative: (1 - t^2) * dL/dout
            self.grad += (1.0 - (t ** 2)) * out.grad
        out._backward = _backward

        return out

    def log(self) -> 'Value':
        # Clamped positive input
        safe_data = max(self.data, 1e-12)
        out = Value(math.log(safe_data), (self,), 'log')

        def _backward():
            # Log derivative: (1 / x) * dL/dout
            self.grad += (1.0 / safe_data) * out.grad
        out._backward = _backward

        return out

    def exp(self) -> 'Value':
        clamped_data = max(min(self.data, 50.0), -50.0)
        e = math.exp(clamped_data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """Executes reverse-mode automatic differentiation across the entire DAG."""
        # Step 1: Topological sort of nodes
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Step 2: Set root gradient (dL/dL = 1.0)
        self.grad = 1.0

        # Step 3: Backpropagate in reverse topological order
        for node in reversed(topo):
            node._backward()


class GradientChecker:
    """Verifies analytical gradients from autograd against central finite difference numerical gradients."""

    @staticmethod
    def verify_derivative(func: Callable[[Value], Value], x_val: float, eps: float = 1e-5) -> Tuple[float, float, float]:
        """Compares analytical derivative with numerical derivative at x_val.
        Returns (analytical_grad, numerical_grad, relative_error).
        """
        # 1. Analytical Gradient via Autograd DAG
        x_node = Value(x_val)
        y_node = func(x_node)
        y_node.backward()
        analytical_grad = x_node.grad

        # 2. Central Finite Difference Numerical Gradient: (f(x + eps) - f(x - eps)) / (2 * eps)
        x_plus = Value(x_val + eps)
        y_plus = func(x_plus).data

        x_minus = Value(x_val - eps)
        y_minus = func(x_minus).data

        numerical_grad = (y_plus - y_minus) / (2.0 * eps)

        # 3. Relative Error Metric
        denominator = max(abs(analytical_grad), abs(numerical_grad), 1e-8)
        relative_error = abs(analytical_grad - numerical_grad) / denominator

        return analytical_grad, numerical_grad, relative_error
