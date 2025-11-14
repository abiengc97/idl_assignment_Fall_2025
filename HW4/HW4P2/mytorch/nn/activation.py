import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        self.A = np.exp(Z - np.max(Z, axis=self.dim, keepdims=True))
        self.A = self.A / np.sum(self.A, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        dim = self.dim
        # Normalize dim to positive index
        if dim < 0:
            dim = len(shape) + dim
        C = shape[dim]
        
        # Reshape input to 2D
        if len(shape) > 2:
            # Step 1: Move dimension to last position
            A_moved = np.moveaxis(self.A, dim, -1)  # Shape: (*, C)
            dLdA_moved = np.moveaxis(dLdA, dim, -1)  # Shape: (*, C)
            
            # Step 2: Flatten remaining dimensions to get 2D tensor
            A_2d = A_moved.reshape(-1, C)  # Shape: (batch_size, C)
            dLdA_2d = dLdA_moved.reshape(-1, C)  # Shape: (batch_size, C)
        else:
            # For 2D or 1D tensors
            A_2d = self.A.reshape(-1, C)
            dLdA_2d = dLdA.reshape(-1, C)
        
        # Step 3: Perform operations on 2D tensor (like HW1P1)
        N = A_2d.shape[0]  # Number of samples
        dLdZ_2d = np.zeros_like(A_2d)  # Initialize output gradient
        
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            # Hint: Jacobian matrix for softmax is a C×C matrix
            J = np.zeros((C, C))
            
            # Fill the Jacobian matrix, please read the writeup for the conditions.
            for m in range(C):
                for n in range(C):
                    J[m, n] = A_2d[i, m] * (1 - A_2d[i, m]) if m == n else -A_2d[i, m] * A_2d[i, n]
            
            # Calculate the derivative of the loss with respect to the i-th input, please read the writeup for it.
            # Hint: How can we use (1×C) and (C×C) to get (1×C) and stack up vertically to give (N×C) derivative matrix?
            dLdZ_2d[i, :] = dLdA_2d[i, :] @ J
        
        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Step 4: Reshape back to shape with dim at end
            dLdZ_moved = dLdZ_2d.reshape(A_moved.shape)
            
            # Step 5: Move dimension back to original position
            dLdZ = np.moveaxis(dLdZ_moved, -1, dim)
        else:
            # For 2D or 1D, reshape back to original shape
            dLdZ = dLdZ_2d.reshape(shape)

        return dLdZ
 

    