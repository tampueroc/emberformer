"""
Test EmberFormer model implementation (Phase 2)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from models import EmberFormer


def test_emberformer_init():
    """Test EmberFormer can be instantiated"""
    print("\n=== Test 1: EmberFormer initialization ===")
    
    try:
        model = EmberFormer(
            d_model=64,
            static_channels=10,
            nhead=4,
            num_layers=2,
            spatial_decoder='unet',  # Use UNet to avoid HuggingFace dependency in test
            patch_size=16,
            max_seq_len=8
        )
        
        print(f"✓ EmberFormer initialized successfully")
        print(f"  - d_model: 64")
        print(f"  - nhead: 4")
        print(f"  - num_layers: 2")
        print(f"  - spatial_decoder: unet")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emberformer_forward():
    """Test EmberFormer forward pass with synthetic data"""
    print("\n=== Test 2: EmberFormer forward pass ===")
    
    try:
        # Create model
        model = EmberFormer(
            d_model=32,  # Smaller for faster test
            static_channels=10,
            nhead=2,
            num_layers=2,
            spatial_decoder='unet',
            patch_size=16,
            max_seq_len=8
        )
        model.eval()
        
        # Create synthetic batch
        B = 2
        N = 25  # 5x5 grid
        T = 4
        Cs = 10
        Gy, Gx = 5, 5
        
        fire_hist = torch.randn(B, N, T)
        wind_hist = torch.randn(B, T, 2)
        static = torch.randn(B, N, Cs)
        valid_t = torch.ones(B, T, dtype=torch.bool)
        
        print(f"✓ Created synthetic batch:")
        print(f"  - fire_hist: {fire_hist.shape}")
        print(f"  - wind_hist: {wind_hist.shape}")
        print(f"  - static: {static.shape}")
        print(f"  - valid_t: {valid_t.shape}")
        
        # Forward pass
        with torch.no_grad():
            logits = model(fire_hist, wind_hist, static, valid_t, grid_shape=(Gy, Gx))
        
        print(f"\n✓ Forward pass successful:")
        print(f"  - Output shape: {logits.shape} (expected: [{B}, 1, {Gy}, {Gx}])")
        
        assert logits.shape == (B, 1, Gy, Gx), f"Shape mismatch: {logits.shape}"
        
        print(f"✓ Output shape correct!")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emberformer_forward_pixels():
    """Test EmberFormer forward_pixels (with unpatchification)"""
    print("\n=== Test 3: EmberFormer forward_pixels ===")
    
    try:
        model = EmberFormer(
            d_model=32,
            static_channels=10,
            nhead=2,
            num_layers=2,
            spatial_decoder='unet',
            patch_size=16,
            max_seq_len=8
        )
        model.eval()
        
        B = 2
        N = 25
        T = 4
        Cs = 10
        Gy, Gx = 5, 5
        P = 16
        
        fire_hist = torch.randn(B, N, T)
        wind_hist = torch.randn(B, T, 2)
        static = torch.randn(B, N, Cs)
        valid_t = torch.ones(B, T, dtype=torch.bool)
        
        # Forward with unpatchification
        with torch.no_grad():
            logits_pixels = model.forward_pixels(
                fire_hist, wind_hist, static, valid_t, grid_shape=(Gy, Gx)
            )
        
        expected_h = Gy * P
        expected_w = Gx * P
        
        print(f"✓ forward_pixels successful:")
        print(f"  - Output shape: {logits_pixels.shape}")
        print(f"  - Expected: [{B}, 1, {expected_h}, {expected_w}]")
        
        assert logits_pixels.shape == (B, 1, expected_h, expected_w), \
            f"Shape mismatch: {logits_pixels.shape} != ({B}, 1, {expected_h}, {expected_w})"
        
        print(f"✓ Unpatchification works correctly!")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_length_sequences():
    """Test EmberFormer handles variable-length temporal sequences"""
    print("\n=== Test 4: Variable-length temporal sequences ===")
    
    try:
        model = EmberFormer(
            d_model=32,
            static_channels=10,
            nhead=2,
            num_layers=2,
            spatial_decoder='unet',
            patch_size=16,
            max_seq_len=8
        )
        model.eval()
        
        B = 3
        N = 25
        T_max = 6  # After left-padding
        Cs = 10
        Gy, Gx = 5, 5
        
        # Simulate variable-length sequences: [3, 5, 2] frames
        fire_hist = torch.zeros(B, N, T_max)
        wind_hist = torch.zeros(B, T_max, 2)
        static = torch.randn(B, N, Cs)
        valid_t = torch.zeros(B, T_max, dtype=torch.bool)
        
        # Sample 0: T=3 (pad first 3)
        fire_hist[0, :, 3:] = torch.randn(N, 3)
        wind_hist[0, 3:, :] = torch.randn(3, 2)
        valid_t[0, 3:] = True
        
        # Sample 1: T=5 (pad first 1)
        fire_hist[1, :, 1:] = torch.randn(N, 5)
        wind_hist[1, 1:, :] = torch.randn(5, 2)
        valid_t[1, 1:] = True
        
        # Sample 2: T=2 (pad first 4)
        fire_hist[2, :, 4:] = torch.randn(N, 2)
        wind_hist[2, 4:, :] = torch.randn(2, 2)
        valid_t[2, 4:] = True
        
        print(f"✓ Created variable-length batch:")
        print(f"  - Sample 0: {valid_t[0].sum().item()}/{T_max} valid timesteps")
        print(f"  - Sample 1: {valid_t[1].sum().item()}/{T_max} valid timesteps")
        print(f"  - Sample 2: {valid_t[2].sum().item()}/{T_max} valid timesteps")
        
        # Forward pass
        with torch.no_grad():
            logits = model(fire_hist, wind_hist, static, valid_t, grid_shape=(Gy, Gx))
        
        print(f"\n✓ Variable-length sequences handled correctly:")
        print(f"  - Output shape: {logits.shape}")
        
        # Check outputs are different (not all zeros/same)
        assert not torch.allclose(logits[0], logits[1]), "Outputs should differ for different inputs"
        
        print(f"✓ Model produces different outputs for different sequence lengths!")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass():
    """Test EmberFormer backward pass (gradient computation)"""
    print("\n=== Test 5: Backward pass ===")
    
    try:
        model = EmberFormer(
            d_model=32,
            static_channels=10,
            nhead=2,
            num_layers=2,
            spatial_decoder='unet',
            patch_size=16,
            max_seq_len=8
        )
        model.train()
        
        B = 2
        N = 25
        T = 4
        Cs = 10
        Gy, Gx = 5, 5
        
        fire_hist = torch.randn(B, N, T)
        wind_hist = torch.randn(B, T, 2)
        static = torch.randn(B, N, Cs)
        valid_t = torch.ones(B, T, dtype=torch.bool)
        target = torch.rand(B, 1, Gy, Gx)
        
        # Forward + loss
        logits = model(fire_hist, wind_hist, static, valid_t, grid_shape=(Gy, Gx))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads, "No gradients computed!"
        
        print(f"✓ Backward pass successful")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Gradients computed: Yes")
        
        # Count parameters with gradients
        grad_params = sum(1 for p in model.parameters() if p.grad is not None)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  - Parameters with gradients: {grad_params}/{total_trainable}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EmberFormer Model Tests (Phase 2)")
    print("="*60)
    
    tests = [
        test_emberformer_init,
        test_emberformer_forward,
        test_emberformer_forward_pixels,
        test_variable_length_sequences,
        test_backward_pass,
    ]
    
    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_fn.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("\n✅ All tests passed! EmberFormer is ready for training.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
