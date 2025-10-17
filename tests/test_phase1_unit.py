"""
Unit tests for Phase 1 implementation (no data required)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from data.collate import collate_tokens_temporal

def test_collate_tokens_temporal_unit():
    """Test collate_tokens_temporal with synthetic data"""
    print("\n=== Unit Test: collate_tokens_temporal ===")
    
    # Create synthetic batch with variable-length sequences
    N = 25  # 5x5 grid
    Cs = 10  # static channels
    
    # Sample 1: T=3
    X1 = {
        "static": torch.randn(N, Cs),
        "fire_hist": torch.randn(3, N),
        "wind_hist": torch.randn(3, 2),
        "valid": torch.ones(N, dtype=torch.bool),
        "meta": torch.tensor([5, 5], dtype=torch.int32),
    }
    y1 = torch.randn(N)
    
    # Sample 2: T=5
    X2 = {
        "static": torch.randn(N, Cs),
        "fire_hist": torch.randn(5, N),
        "wind_hist": torch.randn(5, 2),
        "valid": torch.ones(N, dtype=torch.bool),
        "meta": torch.tensor([5, 5], dtype=torch.int32),
    }
    y2 = torch.randn(N)
    
    # Sample 3: T=2
    X3 = {
        "static": torch.randn(N, Cs),
        "fire_hist": torch.randn(2, N),
        "wind_hist": torch.randn(2, 2),
        "valid": torch.ones(N, dtype=torch.bool),
        "meta": torch.tensor([5, 5], dtype=torch.int32),
    }
    y3 = torch.randn(N)
    
    # Sample 4: T=6
    X4 = {
        "static": torch.randn(N, Cs),
        "fire_hist": torch.randn(6, N),
        "wind_hist": torch.randn(6, 2),
        "valid": torch.ones(N, dtype=torch.bool),
        "meta": torch.tensor([5, 5], dtype=torch.int32),
    }
    y4 = torch.randn(N)
    
    batch = [(X1, y1), (X2, y2), (X3, y3), (X4, y4)]
    
    print(f"✓ Created synthetic batch with T=[3, 5, 2, 6]")
    
    # Collate
    X_batch, y_batch = collate_tokens_temporal(batch)
    
    print(f"\n✓ Collated batch shapes:")
    print(f"  - static: {X_batch['static'].shape} (expected: [4, 25, 10])")
    print(f"  - fire_hist: {X_batch['fire_hist'].shape} (expected: [4, 25, 6])")
    print(f"  - wind_hist: {X_batch['wind_hist'].shape} (expected: [4, 6, 2])")
    print(f"  - valid: {X_batch['valid'].shape} (expected: [4, 25])")
    print(f"  - valid_t: {X_batch['valid_t'].shape} (expected: [4, 6])")
    print(f"  - y: {y_batch.shape} (expected: [4, 25])")
    
    # Check shapes
    B, N_out, T_max = X_batch['fire_hist'].shape
    assert B == 4, f"Batch size mismatch: {B}"
    assert N_out == N, f"N mismatch: {N_out} vs {N}"
    assert T_max == 6, f"T_max should be 6, got {T_max}"
    
    # Check valid_t mask
    valid_counts = X_batch['valid_t'].sum(dim=1).tolist()
    expected_counts = [3, 5, 2, 6]
    
    print(f"\n✓ Valid timesteps per sample:")
    for i, (got, expected) in enumerate(zip(valid_counts, expected_counts)):
        print(f"    Sample {i}: {int(got)}/{T_max} (expected: {expected})")
        assert int(got) == expected, f"Sample {i}: expected {expected} valid timesteps, got {int(got)}"
    
    # Verify left-padding pattern
    print(f"\n✓ Verifying left-padding pattern:")
    for i in range(B):
        valid_t = X_batch['valid_t'][i]
        first_true_idx = torch.where(valid_t)[0]
        if len(first_true_idx) > 0:
            first_true = first_true_idx[0].item()
            # Check that all False come before True
            if first_true > 0:
                assert not valid_t[:first_true].any(), f"Sample {i}: padding should be contiguous at start"
            # Check that all True come after first True
            last_true = first_true_idx[-1].item()
            n_true = len(first_true_idx)
            expected_n_true = last_true - first_true + 1
            assert n_true == expected_n_true, f"Sample {i}: True values should be contiguous"
            print(f"    Sample {i}: padding=[0:{first_true}], valid=[{first_true}:{last_true+1}] ✓")
    
    # Verify fire_hist padding
    print(f"\n✓ Verifying fire_hist content alignment:")
    # Sample 0 should have T=3, so first 3 positions should be zeros
    assert X_batch['fire_hist'][0, 0, :3].abs().sum() == 0, "Sample 0 padding should be zeros"
    assert X_batch['fire_hist'][0, 0, 3:].abs().sum() > 0, "Sample 0 valid data should be non-zero"
    
    # Sample 2 should have T=2, so first 4 positions should be zeros
    assert X_batch['fire_hist'][2, 0, :4].abs().sum() == 0, "Sample 2 padding should be zeros"
    assert X_batch['fire_hist'][2, 0, 4:].abs().sum() > 0, "Sample 2 valid data should be non-zero"
    
    print(f"    Sample 0: first 3 positions are zero (padding) ✓")
    print(f"    Sample 2: first 4 positions are zero (padding) ✓")
    
    print(f"\n✅ All assertions passed!")
    return True


def test_collate_same_length():
    """Test collate when all sequences have same length"""
    print("\n=== Unit Test: collate_tokens_temporal (same length) ===")
    
    N = 25
    Cs = 10
    T = 4
    B = 3
    
    batch = []
    for _ in range(B):
        X = {
            "static": torch.randn(N, Cs),
            "fire_hist": torch.randn(T, N),
            "wind_hist": torch.randn(T, 2),
            "valid": torch.ones(N, dtype=torch.bool),
            "meta": torch.tensor([5, 5], dtype=torch.int32),
        }
        y = torch.randn(N)
        batch.append((X, y))
    
    print(f"✓ Created batch with all sequences T={T}")
    
    X_batch, y_batch = collate_tokens_temporal(batch)
    
    print(f"✓ Collated shapes: fire_hist={X_batch['fire_hist'].shape}")
    
    # All should be valid
    assert X_batch['valid_t'].all(), "All timesteps should be valid when sequences have same length"
    assert X_batch['fire_hist'].shape == (B, N, T), f"Shape mismatch"
    
    print(f"✓ All {B} samples have {T}/{T} valid timesteps")
    print(f"✅ Test passed!")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Phase 1 Unit Tests (no data required)")
    print("="*60)
    
    tests = [
        test_collate_tokens_temporal_unit,
        test_collate_same_length,
    ]
    
    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_fn.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("\n✅ All unit tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
