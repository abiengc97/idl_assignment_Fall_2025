#!/usr/bin/env python3
"""
Fix TorchSummaryX Error

This script fixes the torchsummaryX import error and undefined frames variable.
"""

def fix_torchsummary_cell():
    """Replace the problematic torchsummaryX cell with a working version"""
    
    fixed_cell = '''# Inspect model architecture and check to verify number of parameters of your network
try:
    # Try to install and import torchsummaryX
    !pip install torchsummaryX==1.1.0 --quiet
    from torchsummaryX import summary
    
    # Create a dummy input tensor for the model
    dummy_input = torch.randn(1, 2*config['context']+1, 28).to(device)
    summary(model, dummy_input)
    
except Exception as e:
    print(f"torchsummaryX failed: {e}")
    try:
        # Fallback to torchsummary
        !pip install torchsummary --quiet
        from torchsummary import summary
        
        # Create a dummy input tensor for the model
        dummy_input = torch.randn(1, 2*config['context']+1, 28).to(device)
        summary(model, dummy_input)
        
    except Exception as e2:
        print(f"torchsummary also failed: {e2}")
        print("Using manual parameter count instead...")
        
        # Manual parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        
        # Print model architecture
        print("\\nModel Architecture:")
        print(model)'''
    
    return fixed_cell

def fix_frames_variable():
    """Create a simple frames variable for testing"""
    
    frames_code = '''# Create a dummy frames tensor for testing
# This is just for demonstration - in real usage, frames would come from your dataloader
frames = torch.randn(1, 2*config['context']+1, 28)  # Shape: (batch_size, context*2+1, features)
print(f"Created dummy frames tensor with shape: {frames.shape}")'''
    
    return frames_code

if __name__ == "__main__":
    print("🔧 Fixing TorchSummaryX Error...")
    print("=" * 50)
    
    print("\n📝 Fixed TorchSummaryX Cell:")
    print("-" * 30)
    print(fix_torchsummary_cell())
    
    print("\n📝 Frames Variable Fix:")
    print("-" * 30)
    print(fix_frames_variable())
    
    print("\n📋 Instructions:")
    print("1. Replace the problematic torchsummaryX cell with the fixed version above")
    print("2. Add the frames variable fix before the model inspection cell")
    print("3. Restart your kernel and run the cells again")
    print("4. The error should be resolved!")
