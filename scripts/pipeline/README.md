# Pipeline Scripts

Core hypervolume analysis pipeline implementing Pais et al. (2020) methodology.

## Pipeline Flow

```
[1] extract_salience.py (utils/)  →  data/salience/
[2] prep_u_space.py               →  data/u_space/
[3] envelope_che.py               →  data/che/
[4] create_danger_map.py          →  data/danger_map/
```

## Scripts

| Script | Purpose |
|--------|---------|
| `prep_u_space.py` | Filter extreme fires, engineer features, PCA → U-space |
| `envelope_che.py` | Bootstrap convex hulls → CHE envelope |
| `create_danger_map.py` | Project landscape to U-space, check envelope membership |
| `hypervolume_pipeline.py` | Orchestrator for running all stages |

## Usage

```bash
# Full pipeline
python scripts/pipeline/hypervolume_pipeline.py all \
    --checkpoint checkpoints/model.pt \
    --data_root ~/data/deep_crown_dataset/organized_spreads

# Individual stages
python scripts/pipeline/prep_u_space.py --input data/salience --output data/u_space
python scripts/pipeline/envelope_che.py --input data/u_space --output data/che
python scripts/pipeline/create_danger_map.py --u_space data/u_space --che data/che --data_root <path>
```

## Key Concepts

- **U-Space**: PCA-reduced environmental feature space (≤5 dims, ≥80% variance)
- **CHE**: Convex Hull Ensemble - robust envelope via bootstrap voting
- **Danger Score**: 0.5 = inside envelope (extreme), 1.0 = far from envelope (safe)
- **Landscape-only**: Wind is excluded (temporal/weather) to identify dangerous **locations** regardless of weather

## Features (8 landscape bands)

| Feature | Band | Description |
|---------|------|-------------|
| `forest` | 0 | Forest type classification |
| `arqueo` | 1 | Archaeological sites |
| `cbd` | 2 | Canopy Bulk Density |
| `cbh` | 3 | Canopy Base Height |
| `elevation` | 4 | Terrain elevation (m) |
| `flora` | 5 | Flora classification |
| `paleo` | 6 | Paleontological sites |
| `urbana` | 7 | Urban area classification |

> **Note:** Wind speed/direction are extracted but excluded from U-space projection.
> This identifies locations with dangerous landscape characteristics, independent of weather.
