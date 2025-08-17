# Emberformer

## Data Contract

This project predicts the next fire state from a variable-length history plus static terrain and wind.

- Fire history: fire → [B, 1, H, W, T] : Binary {0,1}. Left-padded along T when histories differ within a batch.
- Static landscape: static → [B, C_static, H, W]: Fixed per scene; normalized to [0,1] per non-binary band; strictly binary bands are not normalized.
- Wind: wind → [B, 2, T]: Channels = (WS, WD) (speed, direction), both normalized to [0,1] using global dataset min/max.
- Valid history mask: valid_tokens → [B, T]: 1 = real timestep, 0 = left-pad. Example for T_max=6, true length 4: [0,0,1,1,1,1].
- Target (next step): target → [B, 1, H, W]: Binary {0,1}. This is the mask for the timestep after the last valid input frame.

Invariants: H and W are identical across fire, static, and target. Timestep spacing is constant across all samples.
