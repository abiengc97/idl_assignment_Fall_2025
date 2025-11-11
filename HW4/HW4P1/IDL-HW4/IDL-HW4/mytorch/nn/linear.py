import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        # Save original shape for reshaping back
        original_shape = A.shape
        in_features = A.shape[-1]
        
        # Reshape to 2D: (batch_size, in_features)
        A_2d = A.reshape(-1, in_features)
        
        # Linear transformation: Z = A @ W.T + b
        # A_2d: (batch_size, in_features)
        # self.W: (out_features, in_features)
        # self.W.T: (in_features, out_features)
        # Result: (batch_size, out_features)
        Z_2d = A_2d @ self.W.T + self.b
        
        # Reshape back to original shape (except last dimension changes to out_features)
        out_features = self.W.shape[0]
        new_shape = original_shape[:-1] + (out_features,)
        Z = Z_2d.reshape(new_shape)
        
        # Store input for backward pass (store the 2D version)
        self.A = A_2d
        self.original_shape = original_shape
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Compute gradients (refer to the equations in the writeup)
        # Reshape dLdZ to 2D: (batch_size, out_features)
        out_features = dLdZ.shape[-1]
        dLdZ_2d = dLdZ.reshape(-1, out_features)
        
        # dLdA = dLdZ @ W
        # dLdZ_2d: (batch_size, out_features)
        # self.W: (out_features, in_features)
        # Result: (batch_size, in_features)
        dLdA_2d = dLdZ_2d @ self.W
        
        # Reshape dLdA back to original input shape
        dLdA = dLdA_2d.reshape(self.original_shape)
        
        # dLdW = dLdZ.T @ A
        # dLdZ_2d.T: (out_features, batch_size)
        # self.A: (batch_size, in_features)
        # Result: (out_features, in_features)
        self.dLdW = dLdZ_2d.T @ self.A
        
        # dLdb = sum over batch dimension
        # dLdZ_2d: (batch_size, out_features)
        # Result: (out_features,)
        self.dLdb = dLdZ_2d.sum(axis=0)
        
        self.dLdA = dLdA
        return dLdA
