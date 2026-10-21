"""
Perceptron Engine & The Non-Linearity / XOR Limitation Proof.
Implements Frank Rosenblatt's (1958) Perceptron algorithm from scratch in NumPy,
evaluates linear separability on logic truth tables, and proves the XOR failure theorem.

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 15-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


class Perceptron:
    """
    Classic Rosenblatt Binary Perceptron classifier implemented from first principles.
    
    Attributes:
        learning_rate (float): Learning rate eta in (0, 1].
        n_epochs (int): Maximum iterations over training dataset.
        random_state (int): Seed for deterministic weight initialization.
        weights (np.ndarray): Weight vector w of shape (n_features, 1).
        bias (float): Scalar threshold intercept b.
        errors_history (List[int]): Number of misclassified samples per epoch.
        is_converged (bool): Flag indicating if weights settled with zero errors.
        converged_epoch (Optional[int]): The epoch number when convergence occurred.
    """
    
    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 50, random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.errors_history: List[int] = []
        self.is_converged: bool = False
        self.converged_epoch: Optional[int] = None
        
    def _step_function(self, z: np.ndarray) -> np.ndarray:
        """Heaviside unit step activation: f(z) = 1 if z >= 0 else 0."""
        return np.where(z >= 0.0, 1, 0)
        
    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Computes linear combination z = X @ w + b."""
        return np.dot(X, self.weights) + self.bias
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels (0 or 1) for input matrix X."""
        return self._step_function(self.net_input(X))
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """
        Trains the perceptron via Rosenblatt's weight update rule:
            w <- w + eta * (y - y_hat) * x
            b <- b + eta * (y - y_hat)
        
        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Binary target labels of shape (n_samples, 1).
            
        Returns:
            self: Fitted Perceptron instance.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize weights with small normal values; bias to 0.0
        self.weights = rng.normal(loc=0.0, scale=0.01, size=(n_features, 1))
        self.bias = 0.0
        self.errors_history = []
        self.is_converged = False
        self.converged_epoch = None
        
        y_col = y.reshape(-1, 1)
        
        for epoch in range(1, self.n_epochs + 1):
            errors = 0
            for xi, target in zip(X, y_col):
                xi_row = xi.reshape(1, -1)
                y_pred = self.predict(xi_row)[0, 0]
                error = target[0] - y_pred
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * xi_row.T
                    self.bias += update
                    errors += 1
            self.errors_history.append(errors)
            
            if errors == 0 and not self.is_converged:
                self.is_converged = True
                self.converged_epoch = epoch
                break
                
        return self
        
    def get_decision_boundary(self, x1_range: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 2D decision boundary line:
            w1 * x1 + w2 * x2 + b = 0 => x2 = -(w1 * x1 + b) / w2
        """
        if self.weights is None or abs(self.weights[1, 0]) < 1e-12:
            return None
        return -(self.weights[0, 0] * x1_range + self.bias) / self.weights[1, 0]


class PerceptronAnalysisSuite:
    """
    Evaluates Perceptron on Logic Gates and provides algebraic/geometric
    proofs of the XOR non-separability theorem.
    """
    
    @staticmethod
    def run_logic_gates_benchmark() -> pd.DataFrame:
        """
        Benchmarks the Single Perceptron across fundamental Boolean operations:
        AND, OR, and XOR.
        """
        gates = {
            "AND": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array([[0], [0], [0], [1]])
            },
            "OR": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array([[0], [1], [1], [1]])
            },
            "XOR": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array([[0], [1], [1], [0]])
            }
        }
        
        results = []
        fitted_models = {}
        
        for gate_name, data in gates.items():
            p = Perceptron(learning_rate=0.1, n_epochs=50, random_state=42)
            p.fit(data["X"], data["y"])
            fitted_models[gate_name] = p
            
            y_pred = p.predict(data["X"])
            acc = np.mean(y_pred == data["y"]) * 100.0
            
            results.append({
                "Logic Gate": gate_name,
                "Linear Separability": "Yes" if gate_name != "XOR" else "No (Non-Linear)",
                "Converged?": "Yes" if p.is_converged else "FAILED (Cycle)",
                "Epochs to Converge": p.converged_epoch if p.is_converged else f"> {p.n_epochs}",
                "Final Accuracy (%)": f"{acc:.1f}%",
                "Final Weights [w1, w2]": f"[{p.weights[0,0]:.3f}, {p.weights[1,0]:.3f}]",
                "Final Bias (b)": f"{p.bias:.3f}"
            })
            
        return pd.DataFrame(results), fitted_models

    @staticmethod
    def get_xor_algebraic_proof() -> str:
        """
        Returns the formal mathematical proof of the Minsky & Papert (1969)
        Perceptron XOR Inseparability Theorem.
        """
        proof = (
            "=========================================================================\n"
            "FORMAL PROOF: Linear Inseparability of the XOR Function (Minsky & Papert)\n"
            "=========================================================================\n"
            "Let a single perceptron compute: f(x1, x2) = 1 if (w1*x1 + w2*x2 + b >= 0) else 0.\n"
            "For the XOR truth table, the weights (w1, w2, b) must simultaneously satisfy:\n\n"
            "  1. Input (0, 0) -> Target 0 :   w1*(0) + w2*(0) + b < 0   ==>   b < 0\n"
            "  2. Input (0, 1) -> Target 1 :   w1*(0) + w2*(1) + b >= 0  ==>   w2 + b >= 0\n"
            "  3. Input (1, 0) -> Target 1 :   w1*(1) + w2*(0) + b >= 0  ==>   w1 + b >= 0\n"
            "  4. Input (1, 1) -> Target 0 :   w1*(1) + w2*(1) + b < 0   ==>   w1 + w2 + b < 0\n\n"
            "DERIVATION OF CONTRADICTION:\n"
            "  Summing inequalities (2) and (3):\n"
            "      (w2 + b) + (w1 + b) >= 0\n"
            "      w1 + w2 + 2b >= 0\n\n"
            "  Since inequality (1) establishes that b < 0, it follows that:\n"
            "      w1 + w2 + b > w1 + w2 + 2b >= 0\n"
            "      w1 + w2 + b > 0\n\n"
            "  However, this strictly CONTRADICTS inequality (4), which requires:\n"
            "      w1 + w2 + b < 0\n\n"
            "CONCLUSION:\n"
            "  No real triplet (w1, w2, b) in R^3 exists to satisfy all four inequalities.\n"
            "  A single linear hyperplane cannot partition the XOR space.\n"
            "  Resolving XOR requires a Multi-Layer Perceptron (MLP) with non-linear activations!\n"
            "========================================================================="
        )
        return proof


if __name__ == "__main__":
    df_res, _ = PerceptronAnalysisSuite.run_logic_gates_benchmark()
    print("=== Perceptron Logic Gates Benchmark ===")
    print(df_res.to_string(index=False))
    print("\n" + PerceptronAnalysisSuite.get_xor_algebraic_proof())
