"""
3D Spot detection model for Piscis.
Uses 3D convolutions instead of 2D.
"""

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Tuple, Union
from functools import partial


class Conv3D(nn.Module):
    """3D Convolutional block with batch norm and activation.
    
    Uses lax.conv_general_dilated for true 3D convolutions.
    """
    
    features: int
    kernel_size: Tuple[int, int, int]
    strides: Tuple[int, int, int] = (1, 1, 1)
    padding: str = 'SAME'
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        from jax import lax
        
        # Input shape: (batch, z, y, x, channels)
        in_features = x.shape[-1]
        
        # Create kernel: (kernel_z, kernel_y, kernel_x, in_features, out_features)
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (*self.kernel_size, in_features, self.features)
        )
        
        # Set up padding
        if self.padding == 'SAME':
            pad_z = (self.kernel_size[0] - 1) // 2
            pad_y = (self.kernel_size[1] - 1) // 2
            pad_x = (self.kernel_size[2] - 1) // 2
            padding = ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x))
        else:
            padding = ((0, 0), (0, 0), (0, 0))
        
        # Apply 3D convolution using lax.conv_general_dilated
        # Input format: (batch, depth, height, width, channels)
        # Kernel format: (depth, height, width, in_channels, out_channels)
        x = lax.conv_general_dilated(
            lhs=x,  # (B, Z, Y, X, C)
            rhs=kernel,  # (Kz, Ky, Kx, Cin, Cout)
            window_strides=self.strides,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),  # 3D convolution format
            feature_group_count=1,
            batch_group_count=1
        )
        
        # Add bias if needed
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            x = x + bias
        
        # Batch norm and activation
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
        x = nn.swish(x)
        
        return x


class ResBlock3D(nn.Module):
    """3D Residual block."""
    
    features: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    downsample: bool = False
    
    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        residual = x
        
        strides = (2, 2, 2) if self.downsample else (1, 1, 1)
        
        # First conv
        x = Conv3D(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=strides
        )(x, train=train)
        
        # Second conv (no activation before residual)
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=(1, 1, 1),
            padding='SAME',
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
        
        # Downsample residual if needed
        if self.downsample or residual.shape[-1] != self.features:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1, 1),
                strides=strides,
                padding='SAME',
                use_bias=False
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(residual)
        
        x = x + residual
        x = nn.swish(x)
        
        return x


class Encoder3D(nn.Module):
    """3D Encoder using residual blocks."""
    
    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> list:
        """Returns list of feature maps at different levels."""
        
        # Initial conv
        x = Conv3D(features=32, kernel_size=(3, 7, 7), strides=(1, 2, 2))(x, train=train)
        
        # Store feature maps at different levels
        features = [x]
        
        # Level 1: 32 features
        x = ResBlock3D(features=32, kernel_size=(3, 3, 3))(x, train=train)
        x = ResBlock3D(features=32, kernel_size=(3, 3, 3))(x, train=train)
        features.append(x)
        
        # Level 2: 64 features (downsample)
        x = ResBlock3D(features=64, kernel_size=(3, 3, 3), downsample=True)(x, train=train)
        x = ResBlock3D(features=64, kernel_size=(3, 3, 3))(x, train=train)
        features.append(x)
        
        # Level 3: 128 features (downsample)
        x = ResBlock3D(features=128, kernel_size=(3, 3, 3), downsample=True)(x, train=train)
        x = ResBlock3D(features=128, kernel_size=(3, 3, 3))(x, train=train)
        features.append(x)
        
        # Level 4: 256 features (downsample)
        x = ResBlock3D(features=256, kernel_size=(3, 3, 3), downsample=True)(x, train=train)
        x = ResBlock3D(features=256, kernel_size=(3, 3, 3))(x, train=train)
        features.append(x)
        
        return features


class Decoder3D(nn.Module):
    """3D Decoder for upsampling feature maps."""
    
    features: int
    
    @nn.compact
    def __call__(self, x: jax.Array, skip: jax.Array, train: bool = True) -> jax.Array:
        from jax import image
        
        # Upsample x to match skip connection size using nearest neighbor
        # Input x: (batch, z1, y1, x1, features)
        # Target skip: (batch, z2, y2, x2, features)
        if x.shape[1:4] != skip.shape[1:4]:
            # Calculate scale factors
            scale_z = skip.shape[1] / x.shape[1]
            scale_y = skip.shape[2] / x.shape[2]
            scale_x = skip.shape[3] / x.shape[3]
            
            # Use resize (which uses nearest neighbor for integer scales)
            # resize expects (batch, z, y, x, features) format
            new_shape = (x.shape[0], skip.shape[1], skip.shape[2], skip.shape[3], x.shape[4])
            x = image.resize(x, new_shape, method='nearest')
        
        # Adjust features to match skip connection if needed
        if x.shape[-1] != skip.shape[-1]:
            x = Conv3D(features=skip.shape[-1], kernel_size=(1, 1, 1))(x, train=train)
        
        # Add skip connection (residual)
        x = x + skip
        
        # Final conv to output features
        x = Conv3D(features=self.features, kernel_size=(3, 3, 3))(x, train=train)
        
        return x


class SpotsModel3D(nn.Module):
    """
    3D Spot detection model.
    
    Attributes
    ----------
    dropout_rate : float
        Dropout rate at skip connections.
    """
    
    dropout_rate: float = 0.2
    
    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        train: bool = True,
        return_style: bool = False
    ) -> Union[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : jax.Array
            Input 3D image with shape (batch, z, y, x, channels)
        train : bool
            Whether in training mode
        return_style : bool
            Whether to return style vector (not implemented for 3D yet)
        
        Returns
        -------
        deltas : jax.Array
            Displacement vectors with shape (batch, z, y, x, 3)
        labels : jax.Array
            Binary labels with shape (batch, z, y, x, 1)
        style : Optional[jax.Array]
            Style vector (if return_style=True)
        """
        
        # Encoder
        encoder = Encoder3D()
        encoder_outputs = encoder(x, train=train)
        
        # Start with the deepest feature map
        x = encoder_outputs[-1]  # Level 4: 256 features
        
        # Decoder with skip connections
        decoder3 = Decoder3D(features=128)
        x = decoder3(x, encoder_outputs[3], train=train)  # Skip from level 3
        
        decoder2 = Decoder3D(features=64)
        x = decoder2(x, encoder_outputs[2], train=train)  # Skip from level 2
        
        decoder1 = Decoder3D(features=32)
        x = decoder1(x, encoder_outputs[1], train=train)  # Skip from level 1
        
        # Final output: 3 features (dz, dy, dx) + 1 label
        x = Conv3D(features=4, kernel_size=(1, 1, 1))(x, train=train)
        
        # Split into deltas (3 channels) and labels (1 channel)
        deltas = x[:, :, :, :, :3]  # (batch, z, y, x, 3)
        labels = nn.sigmoid(x[:, :, :, :, 3:4])  # (batch, z, y, x, 1)
        
        if return_style:
            # For now, return None for style (could be implemented later)
            return deltas, labels, None
        else:
            return deltas, labels


def round_input_size_3d(input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Round 3D SpotsModel input size to be compatible with downsampling.
    
    Parameters
    ----------
    input_size : Tuple[int, int, int]
        Input size (z, y, x).
    
    Returns
    -------
    rounded_input_size : Tuple[int, int, int]
        Rounded input size (z, y, x).
    """
    # The model downsamples by 2^3 = 8 in spatial dimensions
    # For z dimension, we downsample less aggressively (by 2^1 = 2)
    z, y, x = input_size
    
    # Round to nearest multiple of 2 for z (single downsampling)
    z_rounded = 2 * ((z + 1) // 2)
    
    # Round to nearest multiple of 8 for y, x (3 downsampling steps)
    y_rounded = 8 * ((y + 4) // 8)
    x_rounded = 8 * ((x + 4) // 8)
    
    return (z_rounded, y_rounded, x_rounded)
