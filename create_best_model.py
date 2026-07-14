# create_best_model.py

from my_ml_lib.nn import Sequential, Linear, ReLU, Module

class MyMLP(Module):
    def __init__(self, n_features, n_classes):
        super().__init__()

        # Define your best MLP architecture
        self.network = Sequential(
            Linear(n_features, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, n_classes)
        )

    def __call__(self, X):
        # Forward pass
        return self.network(X)

    def __repr__(self):
        # Print network structure neatly
        return f"{self.__class__.__name__}(\n{repr(self.network)}\n)"


def initialize_best_model():
    """
    This function MUST return an instance of your best model's architecture.
    It must match the architecture used when saving 'best_model.npz'.
    
    The autograder will call this function as:
        model = initialize_best_model()
        model.load_state_dict('saved_models/best_model.npz')
    """
    # 784 = 28*28 input features for FashionMNIST, 10 output classes
    model = MyMLP(n_features=784, n_classes=10)
    return model


# Optional: Quick check (runs under 5 mins)
if __name__ == "__main__":
    model = initialize_best_model()
    print(model)

