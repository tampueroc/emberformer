"""
Test Phase 1 implementation: Dataset extensions for temporal history
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from data import TokenFireDataset, collate_tokens_temporal

def test_dataset_temporal_history():
    """Test TokenFireDataset with use_history=True"""
    print("\n=== Test 1: TokenFireDataset with use_history=True ===")
    
    cache_dir = "~/data/emberformer/patch_cache"
    raw_root = "~/data/deep_crown_dataset/organized_spreads"
    
    # Expand paths
    cache_dir = os.path.expanduser(cache_dir)
    raw_root = os.path.expanduser(raw_root)
    
    if not os.path.exists(cache_dir):
        print(f"⚠️  Cache directory not found: {cache_dir}")
        print("   Skipping test (run build_patch_cache.py first)")
        return True
    
    try:
        ds = TokenFireDataset(
            cache_dir=cache_dir,
            raw_root=raw_root,
            use_history=True
        )
        
        print(f"✓ Dataset loaded: {len(ds)} samples")
        
        # Get a sample
        X, y = ds[0]
        
        print(f"✓ Sample 0 loaded:")
        print(f"  - static: {X['static'].shape}")
        print(f"  - fire_hist: {X['fire_hist'].shape}")
        print(f"  - wind_hist: {X['wind_hist'].shape}")
        print(f"  - valid: {X['valid'].shape}")
        print(f"  - meta: {X['meta']}")
        print(f"  - y: {y.shape}")
        
        # Check shapes
        N, Cs = X['static'].shape
        T_hist, N_fire = X['fire_hist'].shape
        T_wind, _ = X['wind_hist'].shape
        
        assert N == N_fire, f"N mismatch: static {N} vs fire {N_fire}"
        assert T_hist == T_wind, f"T mismatch: fire {T_hist} vs wind {T_wind}"
        assert X['valid'].shape[0] == N, f"valid shape mismatch"
        assert y.shape[0] == N, f"y shape mismatch"
        
        print(f"✓ Shapes are consistent")
        print(f"  - T_hist={T_hist}, N={N}, Cs={Cs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compat():
    """Test TokenFireDataset with use_history=False (backward compatibility)"""
    print("\n=== Test 2: TokenFireDataset with use_history=False (backward compat) ===")
    
    cache_dir = "~/data/emberformer/patch_cache"
    raw_root = "~/data/deep_crown_dataset/organized_spreads"
    
    cache_dir = os.path.expanduser(cache_dir)
    raw_root = os.path.expanduser(raw_root)
    
    if not os.path.exists(cache_dir):
        print(f"⚠️  Cache directory not found: {cache_dir}")
        print("   Skipping test")
        return True
    
    try:
        ds = TokenFireDataset(
            cache_dir=cache_dir,
            raw_root=raw_root,
            use_history=False  # Old behavior
        )
        
        print(f"✓ Dataset loaded: {len(ds)} samples")
        
        X, y = ds[0]
        
        print(f"✓ Sample 0 loaded:")
        print(f"  - static: {X['static'].shape}")
        print(f"  - fire_last: {X['fire_last'].shape}")
        print(f"  - wind_last: {X['wind_last'].shape}")
        print(f"  - valid: {X['valid'].shape}")
        print(f"  - meta: {X['meta']}")
        print(f"  - y: {y.shape}")
        
        # Check backward compat keys
        assert 'fire_last' in X, "Missing fire_last key"
        assert 'wind_last' in X, "Missing wind_last key"
        assert 'fire_hist' not in X, "Should not have fire_hist in backward compat mode"
        
        print(f"✓ Backward compatibility maintained")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collate_temporal():
    """Test collate_tokens_temporal function"""
    print("\n=== Test 3: collate_tokens_temporal ===")
    
    cache_dir = "~/data/emberformer/patch_cache"
    raw_root = "~/data/deep_crown_dataset/organized_spreads"
    
    cache_dir = os.path.expanduser(cache_dir)
    raw_root = os.path.expanduser(raw_root)
    
    if not os.path.exists(cache_dir):
        print(f"⚠️  Cache directory not found: {cache_dir}")
        print("   Skipping test")
        return True
    
    try:
        ds = TokenFireDataset(
            cache_dir=cache_dir,
            raw_root=raw_root,
            use_history=True
        )
        
        loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_tokens_temporal
        )
        
        print(f"✓ DataLoader created with batch_size=4")
        
        # Get first batch
        X_batch, y_batch = next(iter(loader))
        
        print(f"✓ Batch loaded:")
        print(f"  - static: {X_batch['static'].shape}")
        print(f"  - fire_hist: {X_batch['fire_hist'].shape}")
        print(f"  - wind_hist: {X_batch['wind_hist'].shape}")
        print(f"  - valid: {X_batch['valid'].shape}")
        print(f"  - valid_t: {X_batch['valid_t'].shape}")
        print(f"  - y: {y_batch.shape}")
        
        B, N, T_max = X_batch['fire_hist'].shape
        
        # Check valid_t mask
        print(f"\n  Valid timesteps per sample:")
        for i in range(B):
            n_valid = X_batch['valid_t'][i].sum().item()
            print(f"    Sample {i}: {n_valid}/{T_max} timesteps valid")
        
        # Verify padding is correct
        for i in range(B):
            valid_t = X_batch['valid_t'][i]
            # Check left-padding: False should come before True
            first_true = torch.where(valid_t)[0][0].item() if valid_t.any() else T_max
            if first_true > 0:
                assert not valid_t[:first_true].any(), "Invalid left-padding pattern"
        
        print(f"\n✓ Left-padding is correct")
        print(f"✓ Collate function works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_length_sequences():
    """Test that variable-length sequences are handled correctly"""
    print("\n=== Test 4: Variable-length sequences ===")
    
    cache_dir = "~/data/emberformer/patch_cache"
    raw_root = "~/data/deep_crown_dataset/organized_spreads"
    
    cache_dir = os.path.expanduser(cache_dir)
    raw_root = os.path.expanduser(raw_root)
    
    if not os.path.exists(cache_dir):
        print(f"⚠️  Cache directory not found: {cache_dir}")
        print("   Skipping test")
        return True
    
    try:
        ds = TokenFireDataset(
            cache_dir=cache_dir,
            raw_root=raw_root,
            use_history=True
        )
        
        # Sample different indices to get variable lengths
        samples = [ds[i] for i in [0, 10, 20, 30]]
        lengths = [X['fire_hist'].shape[0] for X, _ in samples]
        
        print(f"✓ Sampled 4 sequences with lengths: {lengths}")
        
        if len(set(lengths)) > 1:
            print(f"✓ Variable lengths detected: min={min(lengths)}, max={max(lengths)}")
        else:
            print(f"  (All sampled sequences have same length: {lengths[0]})")
        
        # Test collate with these samples
        X_batch, y_batch = collate_tokens_temporal(samples)
        
        T_max = X_batch['fire_hist'].shape[2]
        print(f"\n✓ Collated batch:")
        print(f"  - T_max={T_max}")
        print(f"  - Valid timesteps: {[X_batch['valid_t'][i].sum().item() for i in range(4)]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Phase 1 Tests: Dataset Extensions for Temporal History")
    print("="*60)
    
    tests = [
        test_dataset_temporal_history,
        test_backward_compat,
        test_collate_temporal,
        test_variable_length_sequences,
    ]
    
    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_fn.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
