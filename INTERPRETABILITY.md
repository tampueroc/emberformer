# EmberFormer Interpretability Analysis

**Goal:** Understand what drives fire spread predictions and identify interactions between terrain, wind, temporal history, and extreme events.

**⚠️ IMPORTANT: Run these analyses AFTER model training converges (typically epoch 10-15)**

**Prerequisites:**
1. ✅ Model trained to convergence (val/f1 > 0.65, early stopping triggered or epoch 15+)
2. ✅ Best model checkpoint saved: `checkpoints/emberformer_best.pt`
3. ✅ Validation set available for analysis

**How to Load Converged Model:**
```python
from models import EmberFormer
import torch

# Load best checkpoint
checkpoint = torch.load('checkpoints/emberformer_best.pt')
model = EmberFormer(...config...)  # Use same config as training
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Val F1: {checkpoint['val_f1']:.3f}")
print(f"Val IoU: {checkpoint['val_iou']:.3f}")
```

---

## Research Questions

### 1. Static-Fire Interactions
**Question:** How do terrain features (slope, elevation, vegetation, fuel) influence fire spread direction and speed?

**Hypotheses:**
- Steep slopes → faster uphill spread
- High fuel load → larger spread areas
- Vegetation type → spread patterns

### 2. Wind-Fire Interactions
**Question:** How does wind direction and speed affect fire propagation?

**Hypotheses:**
- Wind direction determines primary spread direction
- Wind speed correlates with spread distance
- Crosswind vs. headwind different spread shapes

### 3. Temporal Dependencies
**Question:** Do longer fire histories improve predictions? Is there a "sweet spot" for sequence length?

**Hypotheses:**
- Longer sequences (T>6) → better long-term trend capture
- Short sequences (T=2-3) → sufficient for local spread
- Diminishing returns after T=8-10

### 4. Extreme Fire Events
**Question:** What factors predict extreme spread events (isochrone area > threshold)?

**Hypotheses:**
- Extreme events correlate with: high wind + steep terrain + dry fuel
- Specific terrain-wind configurations create "fire storms"
- Temporal acceleration patterns (spread increasing over time)

---

## Analysis Methods

## 1. Temporal Attention Visualization

### What to Analyze
Temporal attention weights show **which past timesteps** the model focuses on for prediction.

### Implementation

**Step 1: Extract Attention Weights**

Add to `models/emberformer.py`:
```python
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, ...):
        # ... existing code ...
        self.store_attention = False  # Flag for saving attentions
        self.attention_weights = []   # Storage
    
    def forward(self, tokens, valid_t):
        # ... existing code ...
        
        if self.store_attention:
            # Extract attention from last layer
            # Note: requires modifying TransformerEncoder to return attentions
            # Alternative: use custom transformer with attention output
            pass
        
        return patch_features
```

**Better Approach:** Use attention rollout or custom transformer layer

```python
# Add attention extraction hook
def save_attention_hook(module, input, output):
    # Store attention weights from MultiheadAttention
    if hasattr(module, 'attn_weights'):
        model.attention_weights.append(module.attn_weights)

# Register hook on transformer layers
for layer in model.temporal_transformer.transformer.layers:
    layer.self_attn.register_forward_hook(save_attention_hook)
```

**Step 2: Visualization Script**

Create `scripts/visualize_attention.py`:
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_temporal_attention(model, sample, output_path):
    """
    Visualize which timesteps the model attends to
    
    Args:
        model: Trained EmberFormer
        sample: (X, y) from dataset
        output_path: Where to save visualization
    """
    model.eval()
    model.temporal_transformer.store_attention = True
    
    X, y = sample
    # Forward pass (stores attention)
    with torch.no_grad():
        logits = model(X['fire_hist'], X['wind_hist'], ...)
    
    # attention_weights: [num_layers, B*N, num_heads, T, T]
    # Average over layers, heads, and patches
    attn = torch.stack(model.attention_weights).mean(dim=(0, 2))  # [B*N, T, T]
    attn_avg = attn.mean(dim=0)  # [T, T] - averaged over all patches
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attn_avg.cpu().numpy(), annot=True, fmt='.2f', 
                xticklabels=[f't-{T-i}' for i in range(T)],
                yticklabels=[f't-{T-i}' for i in range(T)])
    ax.set_title('Temporal Attention Weights')
    ax.set_xlabel('Attending to timestep')
    ax.set_ylabel('Query timestep')
    plt.savefig(output_path)
    
    return attn_avg

def plot_attention_by_sequence_length(model, dataset, T_bins=[2, 4, 6, 8, 10]):
    """
    Compare attention patterns for short vs long sequences
    
    Shows: Do longer sequences attend to early history or just recent frames?
    """
    results = {}
    
    for T_min, T_max in zip(T_bins[:-1], T_bins[1:]):
        # Find samples with T in range
        samples = [ds[i] for i in range(len(ds)) 
                   if T_min <= ds[i][0]['fire_hist'].shape[0] < T_max]
        
        # Get average attention for this T range
        attns = [visualize_temporal_attention(model, s, ...) for s in samples[:10]]
        avg_attn = torch.stack(attns).mean(dim=0)
        
        results[f'T={T_min}-{T_max}'] = avg_attn
    
    # Plot comparison
    fig, axes = plt.subplots(1, len(results), figsize=(20, 4))
    for ax, (label, attn) in zip(axes, results.items()):
        sns.heatmap(attn.cpu().numpy(), ax=ax, cmap='viridis')
        ax.set_title(label)
    
    plt.savefig('attention_by_sequence_length.png')
```

**Step 3: Analysis Questions**

- **Q1:** Do long sequences (T>8) attend to early frames (t-8, t-7) or just recent (t-1, t-2)?
  - **If early:** Model captures long-term trends (wind shifts, fuel depletion)
  - **If recent:** Model only needs short-term context
  
- **Q2:** Is there a "sweet spot" for T?
  - Plot: Prediction accuracy vs. sequence length
  - Expected: Performance improves T=2→6, plateaus T>8

---

## 2. Static-Fire Interaction Analysis

### Feature Importance via Ablation

**Method:** Remove or shuffle static features and measure impact

Create `scripts/analyze_static_importance.py`:
```python
def static_feature_ablation(model, val_loader, static_channels):
    """
    Test impact of each static channel on predictions
    
    Args:
        static_channels: ['elevation', 'slope', 'aspect', 'fuel_load', ...]
    
    Returns:
        importance_scores: dict mapping channel → F1 drop when removed
    """
    # Baseline: all features
    baseline_f1 = evaluate_model(model, val_loader)
    
    importance = {}
    for i, channel_name in enumerate(static_channels):
        # Create modified dataset with channel i zeroed out
        def ablate_channel_i(batch):
            X, y = batch
            X['static'][:, :, i] = 0  # Zero out channel i
            return X, y
        
        # Evaluate without this channel
        f1_without = evaluate_model(model, val_loader, transform=ablate_channel_i)
        
        # Importance = drop in F1
        importance[channel_name] = baseline_f1 - f1_without
    
    return importance

# Usage
importance = static_feature_ablation(model, val_loader, 
                                     ['elevation', 'slope', 'aspect', 'fuel'])

# Plot
plt.barh(list(importance.keys()), list(importance.values()))
plt.xlabel('F1 Drop When Removed')
plt.title('Static Feature Importance')
plt.savefig('static_importance.png')
```

### Spatial Correlation Analysis

**Question:** Which terrain types have highest/lowest spread rates?

```python
def analyze_terrain_spread_correlation(dataset, predictions):
    """
    Correlate static features with fire spread magnitude
    
    For each sample:
        - Extract static features at fire boundary
        - Measure spread distance (isochrone area change)
        - Compute correlation
    """
    results = []
    
    for (X, y), pred in zip(dataset, predictions):
        # Get fire boundary pixels
        fire_boundary = get_fire_perimeter(pred)
        
        # Extract static features at boundary
        static_at_boundary = X['static'][fire_boundary]  # [n_boundary, Cs]
        
        # Measure spread (isochrone area)
        spread_area = y.sum() - X['fire_hist'][-1].sum()
        
        results.append({
            'elevation': static_at_boundary[:, 0].mean(),
            'slope': static_at_boundary[:, 1].mean(),
            'spread_area': spread_area,
            'spread_rate': spread_area / (spread_area + 1e-6)
        })
    
    df = pd.DataFrame(results)
    
    # Correlation matrix
    correlations = df.corr()['spread_area'].sort_values(ascending=False)
    
    print("Terrain-Spread Correlations:")
    print(correlations)
    
    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feature in zip(axes.flat, ['elevation', 'slope', 'aspect', 'fuel']):
        ax.scatter(df[feature], df['spread_area'], alpha=0.3)
        ax.set_xlabel(feature)
        ax.set_ylabel('Spread Area')
        ax.set_title(f'{feature} vs Spread')
    
    plt.savefig('terrain_spread_correlations.png')
```

### Directional Spread Analysis

**Question:** Does fire spread uphill (following slope aspect)?

```python
def analyze_directional_spread(dataset):
    """
    Measure fire spread direction vs terrain slope direction
    
    Answers: Does fire preferentially spread uphill?
    """
    for X, y in dataset:
        # Get fire at t and t+1
        fire_t = X['fire_hist'][-1]  # [N]
        fire_t1 = y  # [N] predicted
        
        # Find new fire pixels
        new_fire = (fire_t1 > 0.5) & (fire_t < 0.5)
        
        # Get slope aspect at new fire locations
        aspect = X['static'][:, 2]  # Assuming channel 2 is aspect
        slope = X['static'][:, 1]   # Assuming channel 1 is slope
        
        aspect_at_new_fire = aspect[new_fire]
        slope_at_new_fire = slope[new_fire]
        
        # Compute spread direction
        # ... vector analysis ...
```

---

## 3. Wind-Fire Interaction Analysis

### Wind Direction Alignment

**Question:** Does fire spread align with wind direction?

Create `scripts/analyze_wind_fire.py`:
```python
def compute_wind_fire_alignment(model, dataset):
    """
    Measure alignment between wind direction and fire spread direction
    
    Returns:
        - alignment_scores: cosine similarity between wind and spread vectors
        - spread_magnitudes: spread distance in wind-aligned vs perpendicular
    """
    results = []
    
    for X, y in dataset:
        # Wind vector (last timestep)
        wind = X['wind_hist'][-1]  # [2] - (ws, wd)
        wind_speed, wind_dir = wind[0], wind[1]
        wind_vector = [torch.cos(wind_dir), torch.sin(wind_dir)]
        
        # Fire spread vector (centroid shift)
        fire_t = X['fire_hist'][-1].reshape(Gy, Gx)
        fire_t1 = y.reshape(Gy, Gx)
        
        centroid_t = compute_centroid(fire_t)
        centroid_t1 = compute_centroid(fire_t1)
        spread_vector = centroid_t1 - centroid_t
        
        # Alignment (cosine similarity)
        alignment = cosine_similarity(wind_vector, spread_vector)
        
        # Spread magnitude
        spread_mag = torch.norm(spread_vector)
        
        results.append({
            'wind_speed': wind_speed.item(),
            'wind_dir': wind_dir.item(),
            'alignment': alignment.item(),
            'spread_magnitude': spread_mag.item(),
            'spread_parallel': spread_mag * alignment,  # Component along wind
            'spread_perpendicular': spread_mag * (1 - abs(alignment)),
        })
    
    df = pd.DataFrame(results)
    
    # Analysis
    print(f"Average wind-fire alignment: {df['alignment'].mean():.3f}")
    print(f"Correlation wind_speed vs spread: {df['wind_speed'].corr(df['spread_magnitude']):.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter: wind speed vs spread magnitude
    axes[0].scatter(df['wind_speed'], df['spread_magnitude'], alpha=0.5)
    axes[0].set_xlabel('Wind Speed')
    axes[0].set_ylabel('Spread Magnitude')
    axes[0].set_title('Wind Speed vs Fire Spread')
    
    # Histogram: alignment distribution
    axes[1].hist(df['alignment'], bins=50)
    axes[1].set_xlabel('Wind-Fire Alignment (-1 to 1)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(0, color='red', linestyle='--', label='Perpendicular')
    axes[1].axvline(df['alignment'].mean(), color='green', linestyle='--', label='Mean')
    axes[1].legend()
    axes[1].set_title('Wind-Fire Alignment Distribution')
    
    plt.savefig('wind_fire_interaction.png')
    
    return df

def stratify_by_wind_conditions(df):
    """
    Stratify analysis by wind conditions
    
    Categories:
        - Calm: wind_speed < 5 km/h
        - Moderate: 5-15 km/h
        - Strong: 15-30 km/h
        - Extreme: >30 km/h
    """
    df['wind_category'] = pd.cut(
        df['wind_speed'], 
        bins=[0, 5, 15, 30, 100],
        labels=['Calm', 'Moderate', 'Strong', 'Extreme']
    )
    
    # Group analysis
    grouped = df.groupby('wind_category').agg({
        'spread_magnitude': ['mean', 'std'],
        'alignment': ['mean', 'std'],
        'spread_parallel': 'mean',
    })
    
    print("\nSpread by Wind Category:")
    print(grouped)
    
    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='spread_magnitude', by='wind_category', ax=ax)
    ax.set_ylabel('Spread Magnitude')
    ax.set_title('Fire Spread Distribution by Wind Conditions')
    plt.savefig('spread_by_wind_category.png')
```

### Attention-Based Wind Importance

**Method:** Check if model attends more to windy timesteps

```python
def analyze_wind_attention(model, dataset):
    """
    Correlate temporal attention with wind speed
    
    Question: Does model pay more attention to timesteps with high wind?
    """
    for sample in dataset:
        X, y = sample
        
        # Get attention weights [T, T] for this sample
        attn = get_attention_weights(model, X)  # Needs implementation above
        
        # Get wind speeds over time
        wind_speeds = X['wind_hist'][:, 0]  # [T]
        
        # For each query timestep, check if attention correlates with wind
        # Take last timestep (predicting t+1)
        attn_from_last = attn[-1, :]  # [T] - attention from last to all previous
        
        correlation = torch.corrcoef(torch.stack([wind_speeds, attn_from_last]))[0, 1]
        
        print(f"Correlation(wind_speed, attention): {correlation:.3f}")
```

---

## 4. Long Temporal Dependency Analysis

### Sequence Length Impact Study

**Question:** Do longer sequences improve predictions? What's the sweet spot?

Create `scripts/analyze_sequence_length.py`:
```python
def evaluate_by_sequence_length(model, dataset):
    """
    Stratify evaluation by sequence length
    
    Bins: T∈[1-2], [3-4], [5-6], [7-8], [9-10], [11+]
    
    Metrics for each bin:
        - F1, IoU, Precision, Recall
        - Spread magnitude prediction error
    """
    bins = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 100)]
    results = {f'T={low}-{high}': [] for low, high in bins}
    
    for i, (X, y) in enumerate(dataset):
        T_hist = X['fire_hist'].shape[0]
        
        # Bin this sample
        for low, high in bins:
            if low <= T_hist <= high:
                # Evaluate
                with torch.no_grad():
                    pred = model(X['fire_hist'], ...)
                
                f1 = compute_f1(pred, y)
                iou = compute_iou(pred, y)
                
                results[f'T={low}-{high}'].append({
                    'f1': f1,
                    'iou': iou,
                    'T_hist': T_hist
                })
                break
    
    # Aggregate
    summary = {}
    for bin_name, samples in results.items():
        if samples:
            summary[bin_name] = {
                'count': len(samples),
                'f1_mean': np.mean([s['f1'] for s in samples]),
                'f1_std': np.std([s['f1'] for s in samples]),
                'iou_mean': np.mean([s['iou'] for s in samples]),
            }
    
    # Plot
    bins_labels = list(summary.keys())
    f1_means = [summary[b]['f1_mean'] for b in bins_labels]
    f1_stds = [summary[b]['f1_std'] for b in bins_labels]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(bins_labels)), f1_means, yerr=f1_stds, 
                 marker='o', capsize=5)
    plt.xticks(range(len(bins_labels)), bins_labels, rotation=45)
    plt.xlabel('Sequence Length Bin')
    plt.ylabel('F1 Score')
    plt.title('Model Performance vs Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.savefig('performance_by_sequence_length.png')
    
    return summary

# Run analysis
summary = evaluate_by_sequence_length(model, val_dataset)

# Print findings
print("\nSequence Length Analysis:")
for bin_name, stats in summary.items():
    print(f"{bin_name}: F1={stats['f1_mean']:.3f}±{stats['f1_std']:.3f} "
          f"IoU={stats['iou_mean']:.3f} (n={stats['count']})")
```

### Temporal Information Gain

**Question:** How much does each additional timestep help?

```python
def incremental_context_experiment(model, sample):
    """
    Given a sequence with T=10 frames, evaluate predictions using:
        - Only t=9 (last frame)
        - t=8,9 (last 2 frames)
        - t=7,8,9 (last 3 frames)
        - ...
        - t=0,1,...,9 (full history)
    
    Shows: Diminishing returns curve
    """
    X, y = sample
    full_history = X['fire_hist']  # [T, N]
    T = full_history.shape[0]
    
    f1_by_context = []
    
    for context_len in range(1, T+1):
        # Use only last `context_len` frames
        X_limited = X.copy()
        X_limited['fire_hist'] = full_history[-context_len:]
        X_limited['wind_hist'] = X['wind_hist'][-context_len:]
        
        # Create valid_t mask
        valid_t = torch.ones(context_len, dtype=torch.bool)
        
        # Predict
        with torch.no_grad():
            pred = model(X_limited['fire_hist'], X_limited['wind_hist'], 
                        X_limited['static'], valid_t, ...)
        
        f1 = compute_f1(pred, y)
        f1_by_context.append(f1)
    
    # Plot
    plt.plot(range(1, T+1), f1_by_context, marker='o')
    plt.xlabel('Context Length (# frames)')
    plt.ylabel('F1 Score')
    plt.title('Information Gain from Longer History')
    plt.grid(True, alpha=0.3)
    plt.axhline(f1_by_context[0], color='red', linestyle='--', 
                label='Single frame baseline')
    plt.legend()
    plt.savefig('incremental_context_f1.png')
    
    # Find sweet spot (elbow point)
    gains = [f1_by_context[i] - f1_by_context[i-1] 
             for i in range(1, len(f1_by_context))]
    sweet_spot = np.argmax(np.array(gains) < 0.01) + 1  # Where gain < 1%
    
    print(f"Sweet spot: T={sweet_spot} (diminishing returns after)")
    
    return f1_by_context
```

---

## 5. Extreme Event Analysis

### Identify Extreme Spread Events

**Definition:** Isochrone area growth > threshold (e.g., 90th percentile)

Create `scripts/analyze_extreme_events.py`:
```python
def identify_extreme_events(dataset, percentile=90):
    """
    Find samples with extreme fire spread
    
    Args:
        percentile: Define "extreme" as top N% of spread events
    
    Returns:
        extreme_samples: List of sample indices with extreme spread
        spread_statistics: Distribution of spread magnitudes
    """
    spread_areas = []
    
    for i, (X, y) in enumerate(dataset):
        # Compute spread area (isochrone change)
        fire_t = X['fire_hist'][-1]  # [N]
        fire_t1 = y  # [N] target
        
        area_t = fire_t.sum()
        area_t1 = fire_t1.sum()
        spread = area_t1 - area_t
        
        spread_areas.append({
            'idx': i,
            'spread_area': spread.item(),
            'fire_t': area_t.item(),
            'fire_t1': area_t1.item(),
            'spread_rate': (spread / (area_t + 1e-6)).item()
        })
    
    df = pd.DataFrame(spread_areas)
    
    # Define extreme threshold
    threshold = df['spread_area'].quantile(percentile / 100)
    
    extreme_mask = df['spread_area'] > threshold
    extreme_indices = df[extreme_mask]['idx'].tolist()
    
    print(f"Extreme events (>{percentile}th percentile):")
    print(f"  Threshold: {threshold:.2f} patches")
    print(f"  Count: {len(extreme_indices)} / {len(df)} ({100*len(extreme_indices)/len(df):.1f}%)")
    print(f"  Mean spread (extreme): {df[extreme_mask]['spread_area'].mean():.2f}")
    print(f"  Mean spread (normal): {df[~extreme_mask]['spread_area'].mean():.2f}")
    
    return extreme_indices, df

def analyze_extreme_event_drivers(dataset, extreme_indices):
    """
    Compare static/wind features in extreme vs normal events
    
    Answers: What drives extreme spread?
    """
    extreme_features = []
    normal_features = []
    
    for i, (X, y) in enumerate(dataset):
        # Extract features
        features = {
            'wind_speed': X['wind_hist'][-1, 0].item(),
            'wind_dir': X['wind_hist'][-1, 1].item(),
            'elevation_mean': X['static'][:, 0].mean().item(),
            'slope_mean': X['static'][:, 1].mean().item(),
            'slope_max': X['static'][:, 1].max().item(),
            'fuel_mean': X['static'][:, 3].mean().item() if X['static'].shape[1] > 3 else 0,
            'T_hist': X['fire_hist'].shape[0],
        }
        
        if i in extreme_indices:
            extreme_features.append(features)
        else:
            normal_features.append(features)
    
    df_extreme = pd.DataFrame(extreme_features)
    df_normal = pd.DataFrame(normal_features)
    
    # Compare distributions
    print("\n=== Feature Comparison: Extreme vs Normal ===")
    for col in df_extreme.columns:
        extreme_mean = df_extreme[col].mean()
        normal_mean = df_normal[col].mean()
        diff_pct = 100 * (extreme_mean - normal_mean) / (normal_mean + 1e-6)
        
        print(f"{col:20s}: extreme={extreme_mean:7.3f}, normal={normal_mean:7.3f}, "
              f"diff={diff_pct:+6.1f}%")
    
    # Statistical tests
    from scipy.stats import mannwhitneyu
    
    print("\n=== Statistical Significance (Mann-Whitney U) ===")
    for col in df_extreme.columns:
        statistic, pvalue = mannwhitneyu(df_extreme[col], df_normal[col])
        sig = '***' if pvalue < 0.001 else ('**' if pvalue < 0.01 else ('*' if pvalue < 0.05 else 'ns'))
        print(f"{col:20s}: p={pvalue:.4f} {sig}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, col in zip(axes.flat, df_extreme.columns):
        ax.hist(df_normal[col], bins=30, alpha=0.5, label='Normal', density=True)
        ax.hist(df_extreme[col], bins=30, alpha=0.5, label='Extreme', density=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{col} Distribution')
    
    plt.tight_layout()
    plt.savefig('extreme_vs_normal_features.png')
    
    return df_extreme, df_normal
```

### Extreme Event Prediction Analysis

**Question:** Does the model predict extreme events well?

```python
def evaluate_extreme_event_prediction(model, dataset, extreme_indices):
    """
    Measure model performance specifically on extreme events
    
    Answers: Is model better/worse at extreme spread predictions?
    """
    extreme_f1 = []
    normal_f1 = []
    
    for i, (X, y) in enumerate(dataset):
        with torch.no_grad():
            pred = model(X['fire_hist'], ...)
        
        f1 = compute_f1(pred, y)
        
        if i in extreme_indices:
            extreme_f1.append(f1)
        else:
            normal_f1.append(f1)
    
    print(f"Model Performance:")
    print(f"  Extreme events: F1={np.mean(extreme_f1):.3f} ± {np.std(extreme_f1):.3f}")
    print(f"  Normal events:  F1={np.mean(normal_f1):.3f} ± {np.std(normal_f1):.3f}")
    
    if np.mean(extreme_f1) < np.mean(normal_f1):
        print("  → Model struggles with extreme events!")
        print("  → Consider: oversampling extremes or weighted loss")
```

---

## 6. Isochrone Growth Analysis

### Measuring Fire Spread Magnitude

**Metric:** Change in fire perimeter area (isochrone expansion)

Create `scripts/measure_isochrone_growth.py`:
```python
def compute_isochrone_metrics(fire_t, fire_t1):
    """
    Compute fire spread metrics from consecutive frames
    
    Args:
        fire_t: [H, W] binary mask at time t
        fire_t1: [H, W] binary mask at time t+1
    
    Returns:
        metrics: dict with spread area, rate, perimeter growth, etc.
    """
    # Areas
    area_t = fire_t.sum()
    area_t1 = fire_t1.sum()
    spread_area = area_t1 - area_t
    
    # Rate
    spread_rate = spread_area / (area_t + 1e-6)
    
    # Perimeter analysis
    from scipy.ndimage import distance_transform_edt
    
    # Fire boundary at t
    boundary_t = fire_t & ~binary_erosion(fire_t)
    
    # New fire (spread region)
    new_fire = fire_t1 & ~fire_t
    
    # Spread distance (from old boundary to new fire)
    if new_fire.any() and boundary_t.any():
        # Distance transform from boundary
        dist_from_boundary = distance_transform_edt(~boundary_t)
        spread_distances = dist_from_boundary[new_fire]
        
        mean_spread_dist = spread_distances.mean()
        max_spread_dist = spread_distances.max()
    else:
        mean_spread_dist = 0
        max_spread_dist = 0
    
    return {
        'area_t': area_t,
        'area_t1': area_t1,
        'spread_area': spread_area,
        'spread_rate': spread_rate,
        'mean_spread_distance': mean_spread_dist,
        'max_spread_distance': max_spread_dist,
        'new_fire_area': new_fire.sum(),
    }

def analyze_spread_determinants(dataset):
    """
    Correlate terrain/wind with spread magnitude
    
    Full analysis: What predicts large spread?
    """
    data = []
    
    for X, y in dataset:
        # Compute spread
        fire_t = X['fire_hist'][-1].reshape(Gy, Gx)
        fire_t1 = y.reshape(Gy, Gx)
        
        # Unpatchify to pixels for accurate measurement
        fire_t_pix = F.interpolate(fire_t.unsqueeze(0).unsqueeze(0), 
                                    scale_factor=16, mode='nearest')[0, 0]
        fire_t1_pix = F.interpolate(fire_t1.unsqueeze(0).unsqueeze(0), 
                                     scale_factor=16, mode='nearest')[0, 0]
        
        metrics = compute_isochrone_metrics(fire_t_pix > 0.5, fire_t1_pix > 0.5)
        
        # Extract features
        features = {
            'wind_speed': X['wind_hist'][-1, 0].item(),
            'wind_dir': X['wind_hist'][-1, 1].item(),
            'slope_mean': X['static'][:, 1].mean().item(),
            'slope_max': X['static'][:, 1].max().item(),
            'elevation_std': X['static'][:, 0].std().item(),  # Terrain roughness
            'T_hist': X['fire_hist'].shape[0],
            **metrics
        }
        
        data.append(features)
    
    df = pd.DataFrame(data)
    
    # Correlation analysis
    correlations = df.corr()['spread_area'].sort_values(ascending=False)
    
    print("\n=== Spread Area Correlations ===")
    print(correlations)
    
    # Multiple regression
    from sklearn.linear_model import LinearRegression
    
    X_features = df[['wind_speed', 'slope_mean', 'slope_max', 'T_hist']]
    y_spread = df['spread_area']
    
    model_reg = LinearRegression()
    model_reg.fit(X_features, y_spread)
    
    print("\n=== Linear Regression Coefficients ===")
    for feature, coef in zip(X_features.columns, model_reg.coef_):
        print(f"{feature:20s}: {coef:+.4f}")
    
    print(f"R² score: {model_reg.score(X_features, y_spread):.3f}")
    
    return df
```

### Extreme Spread Threshold Analysis

**Question:** What constitutes "extreme" spread for your data?

```python
def define_extreme_spread_thresholds(dataset):
    """
    Establish thresholds for different severity levels
    
    Creates: percentile-based thresholds
    """
    spread_areas = []
    spread_rates = []
    
    for X, y in dataset:
        fire_t = X['fire_hist'][-1].sum()
        fire_t1 = y.sum()
        spread = fire_t1 - fire_t
        rate = spread / (fire_t + 1e-6)
        
        spread_areas.append(spread.item())
        spread_rates.append(rate.item())
    
    spread_areas = np.array(spread_areas)
    spread_rates = np.array(spread_rates)
    
    # Define severity levels
    thresholds = {
        'Minor': np.percentile(spread_areas, 25),
        'Moderate': np.percentile(spread_areas, 50),
        'Significant': np.percentile(spread_areas, 75),
        'Major': np.percentile(spread_areas, 90),
        'Extreme': np.percentile(spread_areas, 95),
        'Catastrophic': np.percentile(spread_areas, 99),
    }
    
    print("\n=== Spread Area Thresholds (patches) ===")
    for level, thresh in thresholds.items():
        count = (spread_areas > thresh).sum()
        print(f"{level:15s}: >{thresh:6.1f} patches (n={count})")
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of spread areas
    axes[0].hist(spread_areas, bins=100, edgecolor='black')
    for level, thresh in thresholds.items():
        axes[0].axvline(thresh, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Spread Area (patches)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Fire Spread Events')
    axes[0].set_yscale('log')
    
    # Histogram of spread rates
    axes[1].hist(spread_rates, bins=100, edgecolor='black')
    axes[1].set_xlabel('Spread Rate (new_area / existing_area)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Spread Rates')
    
    plt.tight_layout()
    plt.savefig('spread_distribution.png')
    
    return thresholds

def case_study_extreme_events(model, dataset, extreme_indices):
    """
    Deep dive into specific extreme events
    
    For each extreme event:
        - Show terrain map
        - Show wind vector
        - Show temporal progression
        - Show prediction vs reality
        - Compute feature attribution
    """
    for idx in extreme_indices[:5]:  # Top 5 extreme events
        X, y = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            pred = model(X['fire_hist'], ...)
        
        # Create multi-panel visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Panel 1: Terrain (elevation)
        plot_grid(axes[0, 0], X['static'][:, 0], title='Elevation')
        
        # Panel 2: Slope
        plot_grid(axes[0, 1], X['static'][:, 1], title='Slope')
        
        # Panel 3: Fire history (last 4 frames)
        for i, t_idx in enumerate([-4, -3, -2, -1]):
            if -t_idx <= X['fire_hist'].shape[0]:
                plot_grid(axes[0, 2+i], X['fire_hist'][t_idx], 
                         title=f't{t_idx}')
        
        # Panel 4-6: Prediction vs Ground Truth
        plot_grid(axes[1, 0], X['fire_hist'][-1], title='Fire at t')
        plot_grid(axes[1, 1], pred, title='Predicted t+1')
        plot_grid(axes[1, 2], y, title='Actual t+1')
        plot_grid(axes[1, 3], (pred > 0.5) & (y < 0.5), title='False Positives', cmap='Reds')
        
        # Add wind arrow
        wind = X['wind_hist'][-1]
        wind_speed, wind_dir = wind[0].item(), wind[1].item()
        axes[1, 0].arrow(Gx/2, Gy/2, 
                        wind_speed * np.cos(wind_dir) * 3,
                        wind_speed * np.sin(wind_dir) * 3,
                        color='cyan', width=0.5, head_width=2)
        
        plt.suptitle(f'Extreme Event {idx}: Spread={...}')
        plt.tight_layout()
        plt.savefig(f'extreme_event_case_study_{idx}.png')
```

---

## 7. Embedding Space Analysis

### Static Feature Embeddings

**Question:** How does the model encode terrain features?

```python
def visualize_static_embeddings(model):
    """
    Extract and visualize learned static terrain embeddings
    
    Shows: What terrain patterns does model learn?
    """
    # Get static embedding layer
    static_embed = model.token_embedder.static_embed  # Linear(Cs → d_model)
    
    # Extract weight matrix [d_model, Cs]
    W = static_embed.weight.detach().cpu()  # [d_model, Cs]
    
    # PCA or t-SNE on weight vectors
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    W_2d = pca.fit_transform(W.T)  # [Cs, 2]
    
    # Plot
    plt.figure(figsize=(8, 6))
    channel_names = ['elevation', 'slope', 'aspect', 'fuel', ...]
    for i, name in enumerate(channel_names):
        plt.scatter(W_2d[i, 0], W_2d[i, 1], s=100)
        plt.text(W_2d[i, 0], W_2d[i, 1], name, fontsize=10)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Static Feature Embedding Space')
    plt.grid(True, alpha=0.3)
    plt.savefig('static_embedding_space.png')
```

### Wind Embedding Analysis

```python
def analyze_wind_embeddings(model, dataset):
    """
    Test if wind embedding captures directional sensitivity
    
    Method: Vary wind direction, measure prediction change
    """
    sample_X, sample_y = dataset[0]
    
    # Test different wind directions (0°, 45°, 90°, ..., 315°)
    directions = np.linspace(0, 2*np.pi, 8, endpoint=False)
    wind_speed = 20  # Fixed speed
    
    predictions = []
    
    for wd in directions:
        X_modified = sample_X.copy()
        # Set all timesteps to same wind direction
        X_modified['wind_hist'][:, 0] = wind_speed
        X_modified['wind_hist'][:, 1] = wd
        
        with torch.no_grad():
            pred = model(X_modified['fire_hist'], ...)
        
        predictions.append(pred)
    
    # Compute spread direction for each prediction
    spread_dirs = [compute_spread_direction(p) for p in predictions]
    
    # Plot: Wind direction vs Predicted spread direction
    plt.figure(figsize=(8, 8))
    plt.polar(directions, spread_dirs, 'o-')
    plt.title('Wind Direction vs Predicted Spread Direction')
    plt.savefig('wind_direction_sensitivity.png')
    
    # Should see positive correlation if model uses wind properly
    correlation = np.corrcoef(directions, spread_dirs)[0, 1]
    print(f"Wind-Spread direction correlation: {correlation:.3f}")
```

---

## 8. Temporal Dependency Visualization

### Attention Maps Over Time

**Question:** Which past frames matter most for prediction?

```python
def create_attention_timeline(model, sample):
    """
    Create visualization showing attention weights over temporal sequence
    
    Output: Heatmap where:
        - Y-axis: spatial patches
        - X-axis: temporal steps
        - Color: attention weight
    """
    X, y = sample
    T = X['fire_hist'].shape[0]
    N = X['fire_hist'].shape[1]
    
    # Get attention: [N, T] - each patch's attention to each timestep
    attn_weights = extract_temporal_attention(model, X)  # [N, T]
    
    # Reshape to grid
    attn_grid = attn_weights.reshape(Gy, Gx, T)
    
    # Plot for each timestep
    fig, axes = plt.subplots(1, T, figsize=(3*T, 4))
    for t in range(T):
        sns.heatmap(attn_grid[:, :, t], ax=axes[t], cmap='hot', 
                    vmin=0, vmax=attn_grid.max())
        axes[t].set_title(f't-{T-t-1}')
    
    plt.suptitle('Spatial Attention to Each Timestep')
    plt.savefig('temporal_attention_spatial.png')

def compare_short_vs_long_sequences(model, dataset):
    """
    Compare what model learns from T=3 vs T=10 sequences
    
    Hypothesis: Long sequences capture wind shifts, short sequences only local
    """
    short_samples = [s for s in dataset if s[0]['fire_hist'].shape[0] <= 3]
    long_samples = [s for s in dataset if s[0]['fire_hist'].shape[0] >= 10]
    
    # Evaluate both
    short_f1 = evaluate(model, short_samples)
    long_f1 = evaluate(model, long_samples)
    
    print(f"Performance by sequence length:")
    print(f"  Short (T≤3): F1={short_f1:.3f}")
    print(f"  Long (T≥10): F1={long_f1:.3f}")
    print(f"  Gain from longer context: {long_f1 - short_f1:.3f}")
```

---

## 9. Interaction Effect Analysis

### Static × Wind Interactions

**Question:** Do specific terrain-wind combinations amplify spread?

```python
def analyze_terrain_wind_interactions(dataset):
    """
    Find terrain-wind configurations that produce extreme spread
    
    Method: 2D binning (slope × wind_speed) with spread magnitude
    """
    data = []
    
    for X, y in dataset:
        slope_mean = X['static'][:, 1].mean().item()
        wind_speed = X['wind_hist'][-1, 0].item()
        
        fire_t = X['fire_hist'][-1].sum()
        fire_t1 = y.sum()
        spread = (fire_t1 - fire_t).item()
        
        data.append({
            'slope': slope_mean,
            'wind_speed': wind_speed,
            'spread': spread
        })
    
    df = pd.DataFrame(data)
    
    # Create 2D heatmap
    slope_bins = np.linspace(df['slope'].min(), df['slope'].max(), 20)
    wind_bins = np.linspace(df['wind_speed'].min(), df['wind_speed'].max(), 20)
    
    # Bin data
    spread_matrix = np.zeros((len(slope_bins)-1, len(wind_bins)-1))
    count_matrix = np.zeros((len(slope_bins)-1, len(wind_bins)-1))
    
    for _, row in df.iterrows():
        s_idx = np.digitize(row['slope'], slope_bins) - 1
        w_idx = np.digitize(row['wind_speed'], wind_bins) - 1
        
        if 0 <= s_idx < len(slope_bins)-1 and 0 <= w_idx < len(wind_bins)-1:
            spread_matrix[s_idx, w_idx] += row['spread']
            count_matrix[s_idx, w_idx] += 1
    
    # Average spread per bin
    avg_spread = np.divide(spread_matrix, count_matrix, 
                          where=count_matrix > 0, out=np.zeros_like(spread_matrix))
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_spread, xticklabels=np.round(wind_bins[:-1], 1),
                yticklabels=np.round(slope_bins[:-1], 1),
                cmap='YlOrRd', cbar_kws={'label': 'Average Spread Area'})
    plt.xlabel('Wind Speed')
    plt.ylabel('Slope')
    plt.title('Fire Spread Magnitude: Slope × Wind Interaction')
    plt.savefig('slope_wind_interaction_heatmap.png')
    
    # Find extreme combinations
    top_combinations = []
    for i in range(len(slope_bins)-1):
        for j in range(len(wind_bins)-1):
            if count_matrix[i, j] > 5:  # At least 5 samples
                top_combinations.append({
                    'slope': slope_bins[i],
                    'wind': wind_bins[j],
                    'avg_spread': avg_spread[i, j],
                    'count': count_matrix[i, j]
                })
    
    top_combinations = sorted(top_combinations, key=lambda x: x['avg_spread'], reverse=True)
    
    print("\n=== Top 10 High-Spread Configurations ===")
    for i, config in enumerate(top_combinations[:10], 1):
        print(f"{i}. Slope={config['slope']:.2f}°, Wind={config['wind']:.1f} km/h: "
              f"Spread={config['avg_spread']:.1f} patches (n={config['count']})")
    
    return top_combinations
```

---

## 10. Model Decision Boundary Analysis

### Gradient-Based Feature Attribution

**Method:** Integrated Gradients to see what input changes affect predictions most

```python
from captum.attr import IntegratedGradients

def compute_feature_attribution(model, sample):
    """
    Use Integrated Gradients to attribute prediction to inputs
    
    Shows: Which input features drive the prediction?
    """
    X, y = sample
    
    # Baseline (all zeros or mean)
    baseline_fire = torch.zeros_like(X['fire_hist'])
    baseline_wind = torch.zeros_like(X['wind_hist'])
    baseline_static = X['static'].mean(dim=0, keepdim=True).expand_as(X['static'])
    
    # Integrated Gradients
    ig = IntegratedGradients(model)
    
    attributions_fire = ig.attribute(
        X['fire_hist'],
        baselines=baseline_fire,
        additional_forward_args=(X['wind_hist'], X['static'], ...)
    )
    
    # Average over spatial dimension to get temporal importance
    temporal_importance = attributions_fire.abs().mean(dim=1)  # [T]
    
    # Plot
    plt.bar(range(len(temporal_importance)), temporal_importance)
    plt.xlabel('Timestep (0=oldest, T-1=most recent)')
    plt.ylabel('Attribution Magnitude')
    plt.title('Temporal Importance for Prediction')
    plt.savefig('temporal_attribution.png')
    
    return attributions_fire
```

---

## 11. Proposed Analysis Pipeline

### Phase 1: Basic Interpretability (Immediate)

**Week 1-2: After model converges**

1. **Sequence Length Analysis** (2 hours)
   - Run: `evaluate_by_sequence_length()`
   - Find sweet spot for T
   - Answer: Do we need T>6?

2. **Feature Importance** (4 hours)
   - Run: `static_feature_ablation()`
   - Identify: Which terrain features matter most?
   - Answer: Is elevation more important than fuel?

3. **Extreme Event Identification** (2 hours)
   - Run: `identify_extreme_events(percentile=90)`
   - Define thresholds
   - Create dataset of extreme cases

### Phase 2: Interaction Analysis (Research)

**Week 3-4: For paper/thesis**

1. **Wind-Fire Analysis** (1 day)
   - Correlation: wind direction vs spread direction
   - Stratification: performance by wind conditions
   - Visualization: wind sensitivity maps

2. **Terrain-Wind Interactions** (1 day)
   - 2D heatmaps: slope × wind → spread
   - Find dangerous configurations
   - Test model's interaction learning

3. **Temporal Attention** (1 day)
   - Extract attention weights
   - Visualize: what history matters?
   - Compare: short vs long sequence attention patterns

### Phase 3: Extreme Event Deep Dive (Publication)

**Week 5-6: Detailed analysis**

1. **Extreme Event Drivers** (2 days)
   - Statistical tests: extreme vs normal
   - Regression analysis: predict extreme events
   - Feature combinations that trigger extremes

2. **Case Studies** (2 days)
   - Select top 10 extreme events
   - Manual analysis with domain experts
   - Visualization suite for each event

3. **Model Failure Analysis** (1 day)
   - Where does model fail on extremes?
   - Underpredict or overpredict?
   - Error analysis by terrain/wind conditions

---

## 12. Implementation Priority

### Immediate (Do Now)
1. ✅ **Sequence length stratification** - Quick, answers sweet spot question
2. ✅ **Define extreme event thresholds** - Needed for downstream analysis
3. ✅ **Basic wind-spread correlation** - Validate model learns physics

### Short-term (Next 2 Weeks)
4. **Feature ablation studies** - Which inputs matter most?
5. **Extreme event identification** - Create subset for deep analysis
6. **Attention visualization** - Understand temporal dependencies

### Long-term (For Publication)
7. **Interaction effect analysis** - Terrain × wind × time
8. **Case study visualizations** - Extreme event documentation
9. **Feature attribution** - Gradient-based interpretation
10. **Physics validation** - Compare model vs known fire behavior

---

## 13. Expected Findings

### Temporal Dependencies

**Prediction:** Sweet spot at T=6-8

**Evidence to collect:**
- F1 score plateaus after T=7-8
- Attention weights decay exponentially (recent frames weighted higher)
- Long sequences (T>10) don't improve much over T=8

**Publication claim:** "Model effectively captures 6-8 timestep fire history, with diminishing returns beyond T=8, suggesting local temporal dynamics dominate long-range trends."

### Static-Fire Interactions

**Prediction:** Slope and fuel are primary drivers

**Evidence to collect:**
- Slope ablation: -15% F1
- Fuel ablation: -12% F1
- Elevation ablation: -3% F1

**Publication claim:** "Terrain slope and fuel load are critical predictors (ΔF1 > 10%), while elevation shows weaker influence, consistent with fire behavior physics."

### Wind-Fire Interactions

**Prediction:** Strong wind → strong alignment, weak wind → terrain-driven

**Evidence to collect:**
- Wind speed >20 km/h: alignment correlation 0.85
- Wind speed <10 km/h: alignment correlation 0.35
- Crosswind spread reduced by 60%

**Publication claim:** "Wind direction strongly determines spread orientation under high wind (>20 km/h, ρ=0.85), while terrain effects dominate in calm conditions."

### Extreme Events

**Prediction:** Extreme spread = high wind + steep terrain + prior acceleration

**Evidence to collect:**
- Extreme events (90th %ile): slope 45% higher, wind 2x stronger
- 80% of extremes have T≥6 (longer sequences)
- Temporal acceleration: spread_t+1 > 1.5 × spread_t

**Publication claim:** "Extreme fire events (>95th percentile spread) are characterized by steep terrain (slope >30°), strong winds (>25 km/h), and temporal acceleration patterns, with model achieving 72% recall on extreme event detection."

---

## 14. Tools & Dependencies

### Required Packages

```toml
# For interpretability analysis
[project.optional-dependencies]
analysis = [
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
    "pandas>=1.5.0",
    "scikit-learn>=1.2.0",
    "scipy>=1.9.0",
    "captum>=0.6.0",        # For gradient attribution
]
```

Install:
```bash
uv pip install matplotlib seaborn pandas scikit-learn scipy captum
```

### Script Structure

```
scripts/
  analysis/
    01_sequence_length_analysis.py
    02_feature_importance.py
    03_wind_fire_interaction.py
    04_extreme_event_detection.py
    05_attention_visualization.py
    06_terrain_wind_interaction.py
    07_case_studies.py
    08_physics_validation.py
```

---

## 15. Validation Against Fire Physics

### Known Fire Behavior Principles

Test if model learns these:

1. **Upslope spread is faster than downslope**
   - Measure: spread rate upslope vs downslope
   - Expected: 2-3x faster upslope

2. **Wind-driven spread dominates in open terrain**
   - Measure: wind alignment in flat vs mountainous
   - Expected: Higher alignment in flat regions

3. **Fire spreads faster in dry fuel**
   - Measure: correlation fuel_moisture vs spread_rate
   - Expected: Negative correlation

4. **Spread accelerates over time (until fuel exhausted)**
   - Measure: spread_t vs spread_t-1 over sequences
   - Expected: Increasing trend in active fires

```python
def validate_fire_physics(model, dataset):
    """
    Test if model predictions align with known fire behavior
    
    Returns: Physics compliance score
    """
    tests = {
        'upslope_faster': test_upslope_preference(model, dataset),
        'wind_alignment': test_wind_following(model, dataset),
        'fuel_correlation': test_fuel_effect(model, dataset),
        'temporal_acceleration': test_spread_acceleration(model, dataset),
    }
    
    print("=== Physics Validation ===")
    for test_name, (passed, score) in tests.items():
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}: {score:.3f}")
    
    overall_score = np.mean([s for _, s in tests.values()])
    print(f"\nOverall physics compliance: {overall_score:.3f}")
    
    return tests
```

---

## 16. Actionable Next Steps

### Immediate (This Week)

**1. Create analysis directory:**
```bash
mkdir -p scripts/analysis
touch scripts/analysis/__init__.py
```

**2. Implement sequence length analysis:**
```bash
# Start with simplest analysis
scripts/analysis/01_sequence_length_analysis.py
```

**3. Run after epoch 10:**
```python
# In train_emberformer.py, after epoch 10:
if ep == 10 and run:
    # Run quick sequence length analysis
    from analysis import evaluate_by_sequence_length
    seq_analysis = evaluate_by_sequence_length(model, val_set)
    # Log to W&B
    for bin_name, stats in seq_analysis.items():
        run.log({f"analysis/{bin_name}/f1": stats['f1_mean']}, step=step)
```

### Short-term (Next 2 Weeks)

**4. Extreme event dataset:**
- Identify top 10% spread events
- Create separate eval set
- Track model performance on extremes

**5. Wind analysis:**
- Implement wind-spread correlation
- Stratify by wind speed categories
- Visualize alignment

**6. Feature importance:**
- Run ablation study on static channels
- Rank by impact on F1
- Report in thesis

### Long-term (Publication)

**7. Attention analysis:**
- Extract and visualize attention
- Compare short/long sequences
- Publication figure

**8. Interaction effects:**
- Terrain × wind heatmaps
- Statistical interaction tests
- Domain expert validation

**9. Case studies:**
- Top 5 extreme events
- Multi-panel visualizations
- Narrative analysis

---

## 17. Expected Research Contributions

### For Thesis

1. **Temporal Dependency Characterization**
   - "First work to quantify optimal history length for fire spread prediction"
   - "Demonstrated sweet spot at T=6-8 frames"

2. **Extreme Event Detection**
   - "Identified terrain-wind configurations predicting 90% of extreme events"
   - "Model achieves 72% recall on catastrophic spread detection"

3. **Interpretable Attention**
   - "Attention weights reveal model focuses on recent fire shape and wind shifts"
   - "Long sequences primarily used when wind direction changes"

### For Publication

1. **Physics-Informed Validation**
   - "Model predictions align with fire behavior principles (upslope spread, wind following)"
   - "Learned embeddings cluster terrain types consistent with fuel models"

2. **Interaction Discovery**
   - "Identified critical slope-wind thresholds for extreme spread"
   - "Demonstrated non-linear interaction: steep terrain + high wind → 3x spread amplification"

3. **Actionable Insights**
   - "Wind speed >25 km/h + slope >30° → 85% probability of extreme event"
   - "Model predictions enable 2-hour advance warning for dangerous spread"

---

## Summary

**Three Key Analysis Tracks:**

1. **Temporal:** Sweet spot analysis, attention visualization, long dependency validation
2. **Spatial:** Terrain importance, wind alignment, interaction effects  
3. **Extreme Events:** Threshold definition, driver identification, prediction validation

**Start Simple:**
- Sequence length stratification (2 hours, immediate insight)
- Extreme event detection (4 hours, creates research subset)
- Wind-spread correlation (2 hours, physics validation)

**Build Up:**
- Attention extraction (requires model modification)
- Feature attribution (needs captum)
- Statistical interaction tests

**Timeline:** 
- Basic interpretability: 1-2 weeks
- Full analysis: 4-6 weeks
- Publication-ready: 8-10 weeks

All analysis will help answer: **"What makes fires spread faster, and can we predict extreme events?"** 🔥
