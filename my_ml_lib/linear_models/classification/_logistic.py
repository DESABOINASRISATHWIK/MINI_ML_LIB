import numpy as np

class LogisticRegression:
    """
    L2-Regularized Logistic Regression using Newton-Raphson (IRLS).

    Parameters
    ----------
    alpha : float
        Regularization strength (λ)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    """

    def __init__(self, alpha=0.0, max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w_ = None
        self.bias_ = None
        self.converged_ = False

     
    def _sigmoid(self, z):
        # Numerically stable sigmoid
        z = np.clip(z, -500, 500)  # prevent overflow
        return 1 / (1 + np.exp(-z))


    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones((n_samples, 1)), X]
        self.w = np.zeros(X_bias.shape[1])  # ✅ Initialize weights properly

        for i in range(self.max_iter):
            z = X_bias @ self.w
            h = self._sigmoid(z)

            # Gradient
            grad = X_bias.T @ (h - y)
            grad[1:] += self.alpha * self.w[1:]  # regularization

        # Vectorized Hessian
            r = h * (1 - h)
            H = X_bias.T @ (r[:, None] * X_bias)
            H[1:, 1:] += self.alpha * np.eye(n_features)
            H += 1e-6 * np.eye(H.shape[0])  # numerical stability

        # Newton-Raphson update
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                print("Warning: Hessian is singular, stopping early.")
                break

            self.w -= delta

            if np.linalg.norm(delta) < self.tol:
                break

        return self


    def predict_proba(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        z = X_bias @ self.w
        h = self._sigmoid(z)
        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.column_stack([1 - h, h])


    def predict(self, X):
        """
        Return binary class predictions (0 or 1).
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

