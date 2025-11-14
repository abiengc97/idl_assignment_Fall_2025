import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    if padded_input.ndim < 2:
        raise ValueError("padded_input must have at least 2 dimensions (N, T, ...).")

    if input_lengths.ndim != 1:
        raise ValueError("input_lengths must be a 1D tensor containing sequence lengths.")

    if padded_input.size(0) != input_lengths.size(0):
        raise ValueError("Batch size of padded_input must match length of input_lengths.")

    device = padded_input.device
    dtype = torch.bool

    T = padded_input.size(1)
    # Shape: (1, T)
    time_indices = torch.arange(T, device=device)
    # Shape: (N, 1)
    lengths = input_lengths.to(device=device)
    mask = time_indices.unsqueeze(0) >= lengths.unsqueeze(1)
    return mask.to(dtype)

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    if padded_input.ndim < 2:
        raise ValueError("padded_input must have at least 2 dimensions (N, T, ...).")

    device = padded_input.device
    T = padded_input.size(1)
    mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
    return mask

