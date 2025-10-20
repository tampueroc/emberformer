"""
Data augmentation for fire spread prediction

Fire physics are rotationally and reflection invariant, so these augmentations are valid.
"""

import torch
import random


class FireAugmentation:
    """
    Augmentation for fire spread data
    
    Valid transformations (preserve fire physics):
    - Rotations: 90°, 180°, 270°
    - Flips: horizontal, vertical
    - Wind direction rotates accordingly
    """
    
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of applying augmentation
        """
        self.p = p
    
    def __call__(self, fire_hist, static, wind, target):
        """
        Apply augmentation to a sample
        
        Args:
            fire_hist: [1, H, W, T] fire history
            static: [C, H, W] static features
            wind: [T, 2] wind (speed, direction)
            target: [1, H, W] target isochrone
        
        Returns:
            Augmented (fire_hist, static, wind, target)
        """
        if random.random() > self.p:
            return fire_hist, static, wind, target
        
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        
        if k > 0:
            # Rotate spatial data
            fire_hist = torch.rot90(fire_hist, k, dims=[1, 2])  # [1, H, W, T]
            static = torch.rot90(static, k, dims=[1, 2])  # [C, H, W]
            target = torch.rot90(target, k, dims=[1, 2])  # [1, H, W]
            
            # Rotate wind direction accordingly
            # k=1: 90° CCW, k=2: 180°, k=3: 270° CCW
            wind_rot = wind.clone()
            wind_rot[:, 1] = wind[:, 1] + (k * torch.pi / 2)  # Add rotation to direction
            wind = wind_rot
        
        # Random flip (with 50% chance)
        if random.random() > 0.5:
            # Horizontal flip
            fire_hist = torch.flip(fire_hist, dims=[2])  # flip width
            static = torch.flip(static, dims=[2])
            target = torch.flip(target, dims=[2])
            
            # Flip wind direction horizontally (negate x-component)
            wind_flip = wind.clone()
            wind_flip[:, 1] = torch.pi - wind[:, 1]  # Reflect around vertical axis
            wind = wind_flip
        
        return fire_hist, static, wind, target


class ValidationAugmentation:
    """No augmentation for validation"""
    def __call__(self, fire_hist, static, wind, target):
        return fire_hist, static, wind, target
