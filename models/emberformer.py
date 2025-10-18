"""
EmberFormer: Temporal Transformer + Spatial Decoder for fire spread prediction

Architecture:
    1. Temporal Token Embedder: Embeds fire, wind, positional, and static features
    2. Temporal Transformer: Per-patch temporal attention over history
    3. Spatial Decoder: Pre-trained or custom decoder for segmentation
    
DINO Architecture:
    1. DINO Spatial Encoder: Extracts spatial features from fire frames
    2. Feature Fusion: Combines fire, static, wind features
    3. Temporal Transformer: Per-patch temporal attention
    4. Simple Spatial Decoder: Lightweight decoder for binary segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # [max_len, d_model]
    
    def forward(self, T: int) -> torch.Tensor:
        """
        Args:
            T: sequence length
        Returns:
            [T, d_model] positional encodings
        """
        if T > self.max_len:
            # Dynamically extend positional encodings if needed
            device = self.pe.device
            pe_new = torch.zeros(T, self.d_model, device=device)
            position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
            
            pe_new[:, 0::2] = torch.sin(position * div_term)
            pe_new[:, 1::2] = torch.cos(position * div_term)
            return pe_new
        
        return self.pe[:T]


class TemporalTokenEmbedder(nn.Module):
    """
    Embeds per-patch, per-timestep tokens from fire, wind, static, and positional info
    
    Args:
        d_model: embedding dimension
        static_channels: number of static terrain channels
        use_wind: whether to include wind embedding
        use_static: whether to include static terrain embedding
    """
    def __init__(
        self, 
        d_model: int, 
        static_channels: int,
        use_wind: bool = True,
        use_static: bool = True,
        max_seq_len: int = 32
    ):
        super().__init__()
        self.d_model = d_model
        self.use_wind = use_wind
        self.use_static = use_static
        
        # Fire token embedding (single scalar -> d_model)
        self.fire_embed = nn.Linear(1, d_model)
        
        # Wind embedding (2D vector -> d_model)
        if use_wind:
            self.wind_embed = nn.Sequential(
                nn.Linear(2, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model)
            )
        
        # Static terrain embedding (Cs -> d_model), broadcast over time
        if use_static:
            self.static_embed = nn.Linear(static_channels, d_model)
        
        # Temporal positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
    def forward(
        self, 
        fire_hist: torch.Tensor,      # [B, N, T]
        wind_hist: torch.Tensor,      # [B, T, 2]
        static: torch.Tensor,          # [B, N, Cs]
    ) -> torch.Tensor:
        """
        Returns:
            tokens: [B, N, T, d_model]
        """
        B, N, T = fire_hist.shape
        
        # Embed fire: [B, N, T, 1] -> [B, N, T, d]
        fire_emb = self.fire_embed(fire_hist.unsqueeze(-1))  # [B, N, T, d]
        
        # Embed wind: [B, T, 2] -> [B, T, d] -> [B, 1, T, d]
        if self.use_wind:
            wind_emb = self.wind_embed(wind_hist).unsqueeze(1)  # [B, 1, T, d]
            wind_emb = wind_emb.expand(-1, N, -1, -1)  # [B, N, T, d]
        else:
            wind_emb = 0
        
        # Embed static: [B, N, Cs] -> [B, N, d] -> [B, N, 1, d]
        if self.use_static:
            static_emb = self.static_embed(static).unsqueeze(2)  # [B, N, 1, d]
            static_emb = static_emb.expand(-1, -1, T, -1)  # [B, N, T, d]
        else:
            static_emb = 0
        
        # Positional encoding: [T, d] -> [1, 1, T, d]
        pos_emb = self.pos_encoding(T).unsqueeze(0).unsqueeze(0)  # [1, 1, T, d]
        
        # Combine all embeddings
        tokens = fire_emb + wind_emb + static_emb + pos_emb  # [B, N, T, d]
        
        return tokens


class TemporalTransformerEncoder(nn.Module):
    """
    Per-patch temporal transformer encoder
    
    Processes each spatial patch independently over time with self-attention
    
    Args:
        d_model: model dimension
        nhead: number of attention heads
        num_layers: number of transformer layers
        dim_feedforward: feedforward network dimension
        dropout: dropout rate
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(
        self,
        tokens: torch.Tensor,           # [B, N, T, d]
        valid_t: torch.Tensor,          # [B, T] temporal validity mask
    ) -> torch.Tensor:
        """
        Returns:
            patch_features: [B, N, d] - representation for each patch
        """
        B, N, T, d = tokens.shape
        
        # Flatten batch and spatial dimensions: [B*N, T, d]
        tokens_flat = tokens.reshape(B * N, T, d)
        
        # Create padding mask for transformer: [B*N, T]
        # TransformerEncoder expects True for positions to IGNORE
        valid_t_expanded = valid_t.unsqueeze(1).expand(-1, N, -1)  # [B, N, T]
        valid_t_flat = valid_t_expanded.reshape(B * N, T)  # [B*N, T]
        padding_mask = ~valid_t_flat  # Invert: True = padding
        
        # Apply transformer
        encoded = self.transformer(
            tokens_flat,
            src_key_padding_mask=padding_mask
        )  # [B*N, T, d]
        
        # Take last valid timestep for each patch
        # Find last valid index per sequence
        last_valid_idx = valid_t_flat.sum(dim=1) - 1  # [B*N]
        last_valid_idx = last_valid_idx.long().clamp(min=0)
        
        # Gather last valid features
        batch_idx = torch.arange(B * N, device=encoded.device)
        patch_features = encoded[batch_idx, last_valid_idx, :]  # [B*N, d]
        
        # Reshape back: [B, N, d]
        patch_features = patch_features.reshape(B, N, d)
        
        return patch_features


class SpatialDecoderUNet(nn.Module):
    """
    Simple UNet-based spatial decoder for patch grid
    
    Takes per-patch features and outputs patch-level logits
    """
    def __init__(self, d_model: int, base_channels: int = 32):
        super().__init__()
        from .baselines import _DoubleConv, _Down, _Up
        
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        
        # Input projection
        self.proj_in = nn.Conv2d(d_model, c1, 1)
        
        # Encoder
        self.inc = _DoubleConv(c1, c1)
        self.down1 = _Down(c1, c2)
        self.down2 = _Down(c2, c3)
        
        # Decoder
        self.up1 = _Up(c3 + c2, c2)
        self.up2 = _Up(c2 + c1, c1)
        
        # Output
        self.outc = nn.Conv2d(c1, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d, Gy, Gx]
        Returns:
            logits: [B, 1, Gy, Gx]
        """
        x = self.proj_in(x)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u2 = self.up1(x3, x2)
        u1 = self.up2(u2, x1)
        
        return self.outc(u1)


class RefinementDecoder(nn.Module):
    """
    Learned upsampling decoder for pixel-precise predictions.
    
    Takes coarse predictions + features and progressively upsamples to pixel resolution.
    Combines spatial decoder output with features for detail-preserving upsampling.
    
    Args:
        d_model: input feature dimension from temporal transformer
        patch_size: upsampling factor (e.g., 8 or 16)
        base_channels: base channel count for decoder layers
    """
    def __init__(self, d_model: int, patch_size: int = 8, base_channels: int = 32):
        super().__init__()
        self.patch_size = patch_size
        
        # Calculate number of upsampling stages (log2 of patch_size)
        import math
        num_stages = int(math.log2(patch_size))
        assert 2 ** num_stages == patch_size, f"patch_size must be power of 2, got {patch_size}"
        
        # Input combines coarse prediction (1 ch) + features (d_model ch)
        self.input_proj = nn.Conv2d(d_model + 1, base_channels * 4, kernel_size=1)
        
        # Progressive upsampling with residual blocks
        layers = []
        in_ch = base_channels * 4
        out_ch = base_channels * 4
        
        for i in range(num_stages):
            # Transposed conv for 2× upsampling
            layers.append(nn.ConvTranspose2d(
                in_ch, out_ch, 
                kernel_size=4, stride=2, padding=1, bias=False
            ))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            
            # Refinement conv to reduce artifacts
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            
            in_ch = out_ch
            out_ch = max(base_channels, out_ch // 2)  # Reduce channels each stage
        
        self.upsample_layers = nn.Sequential(*layers)
        
        # Final 1×1 conv to prediction
        self.output_conv = nn.Conv2d(in_ch, 1, kernel_size=1)
    
    def forward(self, features: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, d_model, Gy, Gx] features from temporal transformer
            coarse_pred: [B, 1, Gy, Gx] coarse prediction from spatial decoder
        Returns:
            logits: [B, 1, H, W] where H=Gy*patch_size, W=Gx*patch_size
        """
        # Concatenate coarse prediction with features
        x = torch.cat([features, coarse_pred], dim=1)  # [B, d_model+1, Gy, Gx]
        x = self.input_proj(x)
        x = self.upsample_layers(x)
        logits = self.output_conv(x)
        return logits


class SpatialDecoderSegFormer(nn.Module):
    """
    SegFormer-based spatial decoder
    
    Uses pre-trained SegFormer decoder from HuggingFace for segmentation.
    Adapts to our patch-level features via projection layers.
    
    Args:
        d_model: input feature dimension from temporal transformer
        pretrained: whether to load pre-trained weights
        model_name: HuggingFace model name
        freeze_backbone: whether to freeze decoder weights initially
    """
    def __init__(
        self, 
        d_model: int,
        pretrained: bool = True,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        freeze_backbone: bool = False
    ):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError:
            raise ImportError(
                "transformers package required for SegFormer. "
                "Install with: pip install transformers"
            )
        
        # Load pre-trained SegFormer
        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=1,  # Binary segmentation
                ignore_mismatched_sizes=True  # Allow different number of output classes
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(model_name, num_labels=1)
            self.segformer = SegformerForSemanticSegmentation(config)
        
        # Get encoder feature dimensions from SegFormer config
        # SegFormer-B0 has encoder features: [32, 64, 160, 256]
        encoder_channels = self.segformer.config.hidden_sizes
        
        # Projection layer to adapt our d_model features to SegFormer's expected input
        # We'll create a "fake" multi-scale feature pyramid from our single-scale features
        self.feature_projections = nn.ModuleList([
            nn.Conv2d(d_model, ch, 1) for ch in encoder_channels
        ])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.segformer.segformer.encoder.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the SegFormer encoder for fine-tuning"""
        for param in self.segformer.segformer.encoder.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d, Gy, Gx] patch features
        Returns:
            logits: [B, 1, Gy, Gx]
        """
        B, d, Gy, Gx = x.shape
        
        # Create multi-scale features by projecting and downsampling
        # This mimics what a hierarchical encoder would produce
        encoder_hidden_states = []
        
        for i, proj in enumerate(self.feature_projections):
            # Project to target channels
            feat = proj(x)
            
            # Downsample for deeper levels (pyramid structure)
            scale_factor = 2 ** i
            if scale_factor > 1:
                feat = F.avg_pool2d(feat, kernel_size=scale_factor, stride=scale_factor)
            
            encoder_hidden_states.append(feat)
        
        # Use SegFormer decode head directly
        # Skip the encoder and provide our features directly to the decoder
        logits = self.segformer.decode_head(encoder_hidden_states)
        
        # Upsample to match input size if needed
        if logits.shape[-2:] != (Gy, Gx):
            logits = F.interpolate(
                logits, 
                size=(Gy, Gx), 
                mode='bilinear', 
                align_corners=False
            )
        
        return logits


class EmberFormer(nn.Module):
    """
    EmberFormer: Temporal transformer + spatial decoder for fire spread prediction
    
    Architecture:
        Input: fire_hist [B,N,T], wind_hist [B,T,2], static [B,N,Cs]
        ↓
        Temporal Token Embedder → [B,N,T,d]
        ↓
        Temporal Transformer (per-patch attention) → [B,N,d]
        ↓
        Reshape to grid → [B,d,Gy,Gx]
        ↓
        Spatial Decoder → [B,1,Gy,Gx]
        ↓
        Unpatchify → [B,1,H,W]
    
    Args:
        d_model: transformer embedding dimension
        static_channels: number of static terrain channels
        nhead: number of attention heads
        num_layers: number of transformer layers
        dim_feedforward: feedforward dimension
        dropout: dropout rate
        spatial_decoder: 'unet' or custom
        spatial_base_channels: base channels for UNet decoder
        patch_size: patch size for unpatchifying to pixels
        use_wind: whether to use wind embeddings
        use_static: whether to use static embeddings
        max_seq_len: maximum sequence length for positional encoding
    """
    def __init__(
        self,
        d_model: int = 64,
        static_channels: int = 10,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        spatial_decoder: Literal['unet', 'segformer'] = 'segformer',
        spatial_base_channels: int = 32,
        segformer_pretrained: bool = True,
        segformer_model: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        freeze_decoder: bool = False,
        patch_size: int = 16,
        use_wind: bool = True,
        use_static: bool = True,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.decoder_type = spatial_decoder
        
        # Token embedder
        self.token_embedder = TemporalTokenEmbedder(
            d_model=d_model,
            static_channels=static_channels,
            use_wind=use_wind,
            use_static=use_static,
            max_seq_len=max_seq_len
        )
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Spatial decoder
        if spatial_decoder == 'unet':
            self.spatial_decoder = SpatialDecoderUNet(
                d_model=d_model,
                base_channels=spatial_base_channels
            )
        elif spatial_decoder == 'segformer':
            self.spatial_decoder = SpatialDecoderSegFormer(
                d_model=d_model,
                pretrained=segformer_pretrained,
                model_name=segformer_model,
                freeze_backbone=freeze_decoder
            )
        else:
            raise ValueError(f"Unknown spatial decoder: {spatial_decoder}")
        
        # Refinement decoder for pixel-precise predictions
        self.refinement_decoder = RefinementDecoder(
            d_model=d_model,
            patch_size=patch_size,
            base_channels=spatial_base_channels
        )
    
    def unfreeze_decoder(self):
        """Unfreeze spatial decoder for fine-tuning (if using SegFormer)"""
        if self.decoder_type == 'segformer':
            self.spatial_decoder.unfreeze_backbone()
    
    def forward(
        self,
        fire_hist: torch.Tensor,    # [B, N, T]
        wind_hist: torch.Tensor,    # [B, T, 2]
        static: torch.Tensor,        # [B, N, Cs]
        valid_t: torch.Tensor,       # [B, T]
        grid_shape: tuple[int, int], # (Gy, Gx)
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, 1, Gy, Gx] patch-level logits
        """
        B, N, T = fire_hist.shape
        Gy, Gx = grid_shape
        
        assert Gy * Gx == N, f"Grid shape {Gy}x{Gx}={Gy*Gx} != N={N}"
        
        # 1. Embed tokens: [B, N, T, d]
        tokens = self.token_embedder(fire_hist, wind_hist, static)
        
        # 2. Temporal transformer: [B, N, d]
        patch_features = self.temporal_transformer(tokens, valid_t)
        
        # 3. Reshape to spatial grid: [B, d, Gy, Gx]
        features_grid = patch_features.permute(0, 2, 1).reshape(B, self.d_model, Gy, Gx)
        
        # 4. Spatial decoder: [B, 1, Gy, Gx]
        logits = self.spatial_decoder(features_grid)
        
        return logits
    
    def forward_pixels(
        self,
        fire_hist: torch.Tensor,
        wind_hist: torch.Tensor,
        static: torch.Tensor,
        valid_t: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Forward pass with learned upsampling to pixel space
        
        Returns:
            logits_pixels: [B, 1, H, W]
        """
        B, N, T = fire_hist.shape
        Gy, Gx = grid_shape
        
        # 1. Embed tokens: [B, N, T, d]
        tokens = self.token_embedder(fire_hist, wind_hist, static)
        
        # 2. Temporal transformer: [B, N, d]
        patch_features = self.temporal_transformer(tokens, valid_t)
        
        # 3. Reshape to spatial grid: [B, d, Gy, Gx]
        features_grid = patch_features.permute(0, 2, 1).reshape(B, self.d_model, Gy, Gx)
        
        # 4. Spatial decoder: coarse prediction [B, 1, Gy, Gx]
        coarse_logits = self.spatial_decoder(features_grid)
        
        # 5. Refinement decoder: learned upsampling to [B, 1, H, W]
        logits_pixels = self.refinement_decoder(features_grid, coarse_logits)
        
        return logits_pixels


# ============================================================================
# DINO-based Architecture Components
# ============================================================================

class DinoSpatialEncoder(nn.Module):
    """
    DINOv2 encoder for extracting spatial features from fire frames
    
    Uses pretrained DINOv2 to extract rich spatial representations.
    Handles grayscale input by projecting to 3 channels.
    
    Args:
        model_name: HuggingFace model identifier
        frozen: whether to freeze DINO weights
        input_channels: number of input channels (1 for fire, 7 for static)
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        frozen: bool = True,
        input_channels: int = 1,
    ):
        super().__init__()
        
        try:
            from transformers import Dinov2Model
        except ImportError:
            raise ImportError(
                "transformers package required for DINO. "
                "Install with: pip install transformers"
            )
        
        # Load pretrained DINO
        self.dino = Dinov2Model.from_pretrained(model_name)
        self.d_dino = self.dino.config.hidden_size  # 384 (small), 768 (base)
        self.patch_size = self.dino.config.patch_size  # 14
        
        # Adapt input channels (DINO expects 3-channel RGB)
        if input_channels != 3:
            self.input_projection = nn.Conv2d(
                input_channels, 3, 
                kernel_size=1, 
                bias=False
            )
            # Initialize to replicate channels
            with torch.no_grad():
                self.input_projection.weight.fill_(1.0 / input_channels)
        else:
            self.input_projection = nn.Identity()
        
        # Freeze DINO if requested
        if frozen:
            for param in self.dino.parameters():
                param.requires_grad = False
        
        self.frozen = frozen
    
    def unfreeze(self):
        """Unfreeze DINO for fine-tuning"""
        for param in self.dino.parameters():
            param.requires_grad = True
        self.frozen = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input images
        Returns:
            features: [B, N, d_dino] patch features (excludes CLS token)
        """
        # Project to 3 channels
        x = self.input_projection(x)  # [B, 3, H, W]
        
        # DINO forward
        outputs = self.dino(x)
        
        # Get patch tokens (exclude CLS token at position 0)
        features = outputs.last_hidden_state[:, 1:, :]  # [B, N, d_dino]
        
        return features


class FeatureFusion(nn.Module):
    """
    Fuse DINO fire features, static features, and wind
    via concatenation and projection
    
    Args:
        d_dino: DINO feature dimension
        d_wind: wind embedding dimension
        d_model: output dimension for transformer
    """
    def __init__(
        self,
        d_dino: int = 384,
        d_wind: int = 32,
        d_model: int = 256,
    ):
        super().__init__()
        
        # Wind embedding
        self.wind_embed = nn.Sequential(
            nn.Linear(2, d_wind),
            nn.LayerNorm(d_wind),
            nn.GELU(),
        )
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_dino + d_dino + d_wind, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
    
    def forward(
        self,
        fire_features: torch.Tensor,    # [B, T, N, d_dino]
        static_features: torch.Tensor,  # [B, N, d_dino]
        wind: torch.Tensor,             # [B, T, 2]
    ) -> torch.Tensor:
        """
        Returns:
            fused: [B, N, T, d_model] - ready for temporal transformer
        """
        B, T, N, _ = fire_features.shape
        
        # Embed wind
        wind_emb = self.wind_embed(wind)  # [B, T, d_wind]
        
        # Broadcast static and wind to match fire shape
        static_expanded = static_features.unsqueeze(1).expand(B, T, N, -1)
        wind_expanded = wind_emb.unsqueeze(2).expand(B, T, N, -1)
        
        # Concatenate
        combined = torch.cat([
            fire_features,
            static_expanded,
            wind_expanded
        ], dim=-1)  # [B, T, N, d_dino + d_dino + d_wind]
        
        # Project to d_model
        fused = self.fusion_proj(combined)  # [B, T, N, d_model]
        
        # Rearrange for transformer: [B, N, T, d_model]
        fused = fused.transpose(1, 2)
        
        return fused


class SimpleSpatialDecoder(nn.Module):
    """
    Lightweight decoder for patch-grid to binary segmentation
    
    Much simpler than SegFormer - appropriate for binary fire prediction task
    with strong features from temporal transformer.
    
    Args:
        d_model: input feature dimension from transformer
        hidden_channels: hidden layer channels
    """
    def __init__(
        self,
        d_model: int = 256,
        hidden_channels: int = 128,
    ):
        super().__init__()
        
        # Simple upsampling path
        self.decoder = nn.Sequential(
            # Stage 1
            nn.Conv2d(d_model, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Stage 2
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            
            # Output head
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model, Gy, Gx]
        Returns:
            logits: [B, 1, Gy, Gx]
        """
        return self.decoder(x)


class EmberFormerDINO(nn.Module):
    """
    EmberFormer with DINO spatial encoding
    
    Architecture:
        Fire frames [B, T, 1, H, W] → DINO → [B, T, N, d_dino]
        Static [B, Cs, H, W] → DINO → [B, N, d_dino]
        Wind [B, T, 2]
        → Fusion → [B, N, T, d_model]
        → Temporal Transformer → [B, N, d_model]
        → Reshape → [B, d_model, Gy, Gx]
        → Spatial Decoder → [B, 1, Gy, Gx]
        → Refinement → [B, 1, H, W]
    
    Args:
        dino_model: HuggingFace DINO model name
        freeze_dino: whether to freeze DINO encoders
        d_model: transformer embedding dimension
        nhead: number of attention heads
        num_layers: number of transformer layers
        dim_feedforward: feedforward dimension
        dropout: dropout rate
        spatial_hidden: hidden channels for spatial decoder
        patch_size: patch size for refinement
        fusion_type: feature fusion strategy
    """
    def __init__(
        self,
        # DINO config
        dino_model: str = "facebook/dinov2-small",
        freeze_dino: bool = True,
        
        # Transformer config
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        
        # Decoder config
        spatial_hidden: int = 128,
        patch_size: int = 16,
        
        # Static channels
        static_channels: int = 7,
    ):
        super().__init__()
        
        # DINO encoders
        self.fire_encoder = DinoSpatialEncoder(
            model_name=dino_model,
            frozen=freeze_dino,
            input_channels=1,  # Grayscale fire
        )
        
        self.static_encoder = DinoSpatialEncoder(
            model_name=dino_model,
            frozen=True,  # Always freeze static encoder
            input_channels=static_channels,
        )
        
        d_dino = self.fire_encoder.d_dino
        
        # Feature fusion
        self.fusion = FeatureFusion(
            d_dino=d_dino,
            d_wind=32,
            d_model=d_model,
        )
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Spatial decoder
        self.spatial_decoder = SimpleSpatialDecoder(
            d_model=d_model,
            hidden_channels=spatial_hidden,
        )
        
        # Refinement decoder
        self.refinement_decoder = RefinementDecoder(
            d_model=d_model,
            patch_size=patch_size,
            base_channels=32,
        )
        
        self.patch_size = patch_size
        self.d_model = d_model
    
    def unfreeze_dino(self):
        """Unfreeze DINO fire encoder for fine-tuning"""
        self.fire_encoder.unfreeze()
    
    def forward(
        self,
        fire_hist: torch.Tensor,   # [B, T, 1, H, W]
        static: torch.Tensor,      # [B, Cs, H, W]
        wind: torch.Tensor,        # [B, T, 2]
        valid_t: torch.Tensor,     # [B, T] temporal validity mask
        target_size: tuple = None, # Optional (H, W) for output size
    ) -> torch.Tensor:
        """
        Args:
            fire_hist: Fire history frames
            static: Static terrain features
            wind: Wind conditions
            valid_t: Temporal validity mask
            target_size: Optional output size (H, W). If None, uses input size.
        
        Returns:
            predictions: [B, 1, H, W]
        """
        B, T, _, H, W = fire_hist.shape
        
        # Use target_size if provided, otherwise use input size
        if target_size is None:
            target_size = (H, W)
        
        # Encode fire frames with DINO
        # Reshape to process all timesteps together
        fire_frames = fire_hist.reshape(B * T, 1, H, W)
        fire_features = self.fire_encoder(fire_frames)  # [B*T, N, d_dino]
        
        # Reshape back
        N = fire_features.shape[1]
        fire_features = fire_features.reshape(B, T, N, -1)
        
        # Encode static terrain (single pass)
        static_features = self.static_encoder(static)  # [B, N, d_dino]
        
        # Fuse features
        fused = self.fusion(
            fire_features=fire_features,
            static_features=static_features,
            wind=wind,
        )  # [B, N, T, d_model]
        
        # Temporal transformer
        temporal_features = self.temporal_transformer(
            fused, 
            valid_t=valid_t
        )  # [B, N, d_model]
        
        # Reshape to grid
        Gy = Gx = int(np.sqrt(N))
        assert Gy * Gx == N, f"N={N} must be square"
        
        grid_features = temporal_features.transpose(1, 2).reshape(B, self.d_model, Gy, Gx)
        
        # Spatial decoder
        patch_logits = self.spatial_decoder(grid_features)  # [B, 1, Gy, Gx]
        
        # Refinement to pixels
        pixel_logits = self.refinement_decoder(
            grid_features,
            patch_logits
        )  # [B, 1, H_out, W_out]
        
        # Resize to target size if needed (DINO patch size != refinement patch size)
        if pixel_logits.shape[-2:] != target_size:
            pixel_logits = F.interpolate(
                pixel_logits,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return pixel_logits
