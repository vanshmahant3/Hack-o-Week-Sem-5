import numpy as np
from typing import List, Union
from autograd import Value

class Neuron:
    """A single artificial neuron with learnable weights, bias, and non-linear activation."""
    def __init__(self, nin: int, nonlin: bool = True, act_fn: str = 'relu'):
        # Xavier / He random initialization
        scale = np.sqrt(2.0 / nin)
        self.w = [Value(np.random.randn() * scale) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin
        self.act_fn = act_fn

    def __call__(self, x: List[Union[Value, float]]) -> Value:
        # Affine combination: z = sum(w_i * x_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if not self.nonlin:
            return act
        if self.act_fn == 'relu':
            return act.relu()
        elif self.act_fn == 'sigmoid':
            return act.sigmoid()
        elif self.act_fn == 'tanh':
            return act.tanh()
        return act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    """A layer of independent neurons receiving identical input vectors."""
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: List[Union[Value, float]]) -> List[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-Layer Perceptron (Neural Network) built on our Autograd DAG."""
    def __init__(self, nin: int, nouts: List[int], act_fns: List[str] = None):
        sz = [nin] + nouts
        if act_fns is None:
            act_fns = ['relu'] * (len(nouts) - 1) + ['sigmoid']
        
        self.layers = []
        for i in range(len(nouts)):
            is_last = (i == len(nouts) - 1)
            act = act_fns[i] if act_fns else ('sigmoid' if is_last else 'relu')
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=True, act_fn=act))

    def __call__(self, x: List[Union[Value, float]]) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        """Zeroes accumulated gradients across all network parameters."""
        for p in self.parameters():
            p.grad = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.05, batch_size: int = 32) -> dict:
        """Trains the neural network using pure backpropagation & Stochastic Gradient Descent (SGD)."""
        n_samples = len(X)
        history = {'loss': [], 'accuracy': [], 'grad_norms': []}

        for epoch in range(1, epochs + 1):
            # Shuffle batch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            correct_preds = 0
            n_batches = int(np.ceil(n_samples / batch_size))

            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                curr_bs = len(X_batch)

                # 1. Forward pass on batch
                self.zero_grad()
                batch_losses = []
                
                for xi, yi in zip(X_batch, y_batch):
                    y_pred = self(xi.tolist())
                    # Binary Cross-Entropy Loss: -[y*log(p) + (1-y)*log(1-p)]
                    if yi == 1:
                        loss_i = -y_pred.log()
                    else:
                        loss_i = -(Value(1.0) - y_pred).log()
                    batch_losses.append(loss_i)

                    pred_class = 1 if y_pred.data >= 0.5 else 0
                    if pred_class == yi:
                        correct_preds += 1

                # Average batch loss
                total_loss = sum(batch_losses, Value(0.0)) / curr_bs
                epoch_loss += total_loss.data * curr_bs

                # 2. Backward pass (Multivariate Chain Rule)
                total_loss.backward()

                # 3. Parameter updates (SGD)
                for p in self.parameters():
                    p.data -= lr * p.grad

            # Track epoch metrics
            avg_loss = epoch_loss / n_samples
            accuracy = correct_preds / n_samples
            total_grad_norm = float(np.sqrt(sum(p.grad ** 2 for p in self.parameters())))

            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            history['grad_norms'].append(total_grad_norm)

        return history
