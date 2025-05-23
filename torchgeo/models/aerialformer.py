"""AerialFormer implementation using TorchGeo's Swin Transformer.

This module implements AerialFormer: Multi-resolution Transformer for
aerial image semantic segmentation, which combines CNN Stem, Swin Transformer
backbone, and Multi-Dilated CNN (MDC) decoder.

Reference:
    Hanyu, T., Yamazaki, K., Tran, M., et al. (2024). AerialFormer: Multi-Resolution
    Transformer for Aerial Image Segmentation. Remote Sensing, 16(16), 2930.
    https://doi.org/10.3390/rs16162930
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformer

from torchgeo.models import Swin_V2_B_Weights, Swin_V2_T_Weights, swin_v2_b, swin_v2_t


class CNNStem(nn.Module):
    """CNN stem for improved low-level feature extraction.

    The CNN Stem is designed to preserve fine-grained details and high-resolution
    features, particularly for tiny objects in aerial imagery. It maintains half
    the spatial resolution of the input image.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB).
        out_channels: Number of output channels (should match Swin embed_dim).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 96) -> None:
        """Initialize CNN Stem.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        # Three-stage CNN with progressively increasing channels
        self.stem = nn.Sequential(
            # Stage 1: Initial feature extraction with spatial reduction
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Stage 2: Feature refinement
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Stage 3: Additional refinement
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Stage 4: Channel adjustment to match backbone
            nn.Conv2d(64, out_channels // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN stem.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, out_channels//2, H//2, W//2).
        """
        return self.stem(x)


class DilatedConvLayer(nn.Module):
    """Dilated Convolution Layer (DCL) component of MDC block.

    Splits input channels and applies different dilation rates to capture
    multi-scale contextual information efficiently.

    Args:
        in_channels: Number of input channels.
        kernel_sizes: Tuple of kernel sizes for each branch.
        dilations: Tuple of dilation rates for each branch.
        paddings: Tuple of padding values for each branch.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_sizes: tuple[int, int, int] = (3, 3, 3),
        dilations: tuple[int, int, int] = (1, 2, 3),
        paddings: tuple[int, int, int] = (1, 2, 3),
    ) -> None:
        """Initialize Dilated Convolution Layer.

        Args:
            in_channels: Number of input channels.
            kernel_sizes: Kernel sizes for each branch.
            dilations: Dilation rates for each branch.
            paddings: Padding values for each branch.
        """
        super().__init__()

        # Split channels evenly with special handling for remainders
        split_channels = self._calculate_channel_splits(in_channels)
        self.split_channels = split_channels

        # Create dilated convolution branches
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    split_channels[i],
                    split_channels[i],
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i],
                    dilation=dilations[i],
                    bias=False,
                )
                for i in range(3)
            ]
        )

    @staticmethod
    def _calculate_channel_splits(channels: int) -> list[int]:
        """Calculate channel splits for 3 branches.

        Args:
            channels: Total number of channels to split.

        Returns:
            List of channel counts for each branch.
        """
        quotient = channels // 3
        remainder = channels % 3

        splits = [quotient] * 3
        if remainder == 1:
            splits[0] += 1
        elif remainder == 2:
            splits[0] += 1
            splits[1] += 1

        return splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilated convolutions with channel splitting.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        # Split input channels
        x_splits = torch.split(x, self.split_channels, dim=1)

        # Apply dilated convolutions to each split
        outputs = [branch(x_split) for branch, x_split in zip(self.branches, x_splits)]

        # Concatenate results
        return torch.cat(outputs, dim=1)


class MDCBlock(nn.Module):
    """Multi-Dilated Convolution (MDC) Block.

    Captures multi-scale contextual information through parallel dilated
    convolutions with different receptive fields. Includes pre/post channel
    mixing for feature refinement.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_sizes: Kernel sizes for dilated convolutions.
        dilations: Dilation rates for parallel branches.
        paddings: Padding values corresponding to dilations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, int, int] = (3, 3, 3),
        dilations: tuple[int, int, int] = (1, 2, 3),
        paddings: tuple[int, int, int] = (1, 2, 3),
    ) -> None:
        """Initialize MDC block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_sizes: Kernel sizes for dilated convolutions.
            dilations: Dilation rates for parallel branches.
            paddings: Padding values corresponding to dilations.
        """
        super().__init__()

        # Pre-channel mixer (1x1 conv for channel adjustment)
        self.pre_mixer = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Dilated convolution layer
        self.dcl = DilatedConvLayer(in_channels, kernel_sizes, dilations, paddings)

        # Post-channel mixer (1x1 conv for fusion and channel adjustment)
        self.post_mixer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MDC block.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        x = self.pre_mixer(x)
        x = self.dcl(x)
        x = self.post_mixer(x)
        return x


class MDCDecoder(nn.Module):
    """Multi-scale Decoder with Multi-Dilated Convolutions.

    Progressive decoder that aggregates multi-scale features through skip
    connections and applies MDC blocks for contextual enhancement at each scale.

    Args:
        in_channels: List of input channel counts from encoder (in forward order).
        num_classes: Number of output classes for segmentation.
        dropout_ratio: Dropout probability for regularization.
    """

    def __init__(
        self, in_channels: list[int], num_classes: int, dropout_ratio: float = 0.1
    ) -> None:
        """Initialize MDC Decoder.

        Args:
            in_channels: List of input channel counts from encoder.
            num_classes: Number of output classes for segmentation.
            dropout_ratio: Dropout probability for regularization.
        """
        super().__init__()

        # Reverse channel list for bottom-up processing
        self.in_channels = list(reversed(in_channels))
        self.num_stages = len(self.in_channels)

        # Define MDC parameters for each decoder stage
        mdc_configs = [
            # Stage 0 (deepest): larger dilations for global context
            {'kernel_sizes': (3, 3, 3), 'dilations': (3, 5, 7), 'paddings': (3, 5, 7)},
            # Stage 1: medium dilations
            {'kernel_sizes': (3, 3, 3), 'dilations': (1, 2, 3), 'paddings': (1, 2, 3)},
            # Stage 2: smaller dilations
            {'kernel_sizes': (3, 3, 3), 'dilations': (1, 2, 3), 'paddings': (1, 2, 3)},
            # Stage 3: minimal dilations for local refinement
            {'kernel_sizes': (3, 3, 3), 'dilations': (1, 1, 1), 'paddings': (1, 1, 1)},
            # Stage 4 (if CNN stem): no dilation for finest details
            {'kernel_sizes': (1, 3, 3), 'dilations': (1, 1, 1), 'paddings': (0, 1, 1)},
        ]

        # Build decoder stages
        self.up_convs = nn.ModuleList()
        self.decode_blocks = nn.ModuleList()

        for idx in range(self.num_stages):
            # Upsampling layer (skip for first stage)
            if idx == 0:
                self.up_convs.append(nn.Identity())
            else:
                self.up_convs.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            self.in_channels[idx - 1],
                            self.in_channels[idx],
                            kernel_size=2,
                            stride=2,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.in_channels[idx]),
                        nn.GELU(),
                    )
                )

            # MDC block with refinement convolution
            in_ch = self.in_channels[idx] * 2 if idx > 0 else self.in_channels[idx]
            self.decode_blocks.append(
                nn.Sequential(
                    MDCBlock(
                        in_channels=in_ch,
                        out_channels=self.in_channels[idx],
                        **mdc_configs[idx],
                    ),
                    nn.Conv2d(
                        self.in_channels[idx],
                        self.in_channels[idx],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.in_channels[idx]),
                    nn.GELU(),
                )
            )

        # Final layers
        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )
        self.conv_seg = nn.Conv2d(self.in_channels[-1], num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale features to segmentation map.

        Args:
            features: List of feature tensors from encoder stages,
                     ordered from low to high resolution.

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        # Reverse features for bottom-up processing
        features = list(reversed(features))

        # Start from deepest features
        x = features[0]
        x = self.decode_blocks[0](x)

        # Progressive upsampling with skip connections
        for idx in range(1, self.num_stages):
            x = self.up_convs[idx](x)
            x = torch.cat([x, features[idx]], dim=1)
            x = self.decode_blocks[idx](x)

        # Final classification
        x = self.dropout(x)
        x = self.conv_seg(x)

        return x


class AerialFormer(nn.Module):
    """AerialFormer: Multi-resolution Transformer for aerial image segmentation.

    Combines a CNN Stem for low-level feature extraction, Swin Transformer
    backbone for hierarchical feature learning, and Multi-Dilated CNN decoder
    for multi-scale context aggregation.

    Args:
        num_classes: Number of segmentation classes.
        in_channels: Number of input image channels.
        backbone: Swin transformer variant ('swin_v2_t' or 'swin_v2_b').
        pretrained: Whether to use pretrained weights.
        pretrained_weights: Specific TorchGeo weights to load (if available).
        decoder_dropout: Dropout ratio in decoder.
        use_cnn_stem: Whether to use CNN stem for low-level features.

    Examples:
        >>> # Basic usage with default settings
        >>> model = AerialFormer(num_classes=10, in_channels=3)
        >>> x = torch.randn(2, 3, 512, 512)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 10, 512, 512])

        >>> # Using TorchGeo pretrained weights for Sentinel-2
        >>> model = AerialFormer(
        ...     num_classes=10,
        ...     in_channels=9,
        ...     backbone='swin_v2_b',
        ...     pretrained_weights='SENTINEL2_MI_MS_SATLAS'
        ... )
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        backbone: str = 'swin_v2_t',
        pretrained: bool = False,
        pretrained_weights: str | None = None,
        decoder_dropout: float = 0.1,
        use_cnn_stem: bool = True,
    ) -> None:
        """Initialize AerialFormer.

        Args:
            num_classes: Number of segmentation classes.
            in_channels: Number of input image channels.
            backbone: Swin transformer variant ('swin_v2_t' or 'swin_v2_b').
            pretrained: Whether to use pretrained weights.
            pretrained_weights: Specific TorchGeo weights to load (if available).
            decoder_dropout: Dropout ratio in decoder.
            use_cnn_stem: Whether to use CNN stem for low-level features.
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_cnn_stem = use_cnn_stem

        # Backbone configuration
        backbone_configs = {
            'swin_v2_t': {'embed_dim': 96, 'depths': [2, 2, 6, 2]},
            'swin_v2_b': {'embed_dim': 128, 'depths': [2, 2, 18, 2]},
        }

        if backbone not in backbone_configs:
            raise ValueError(f'Unsupported backbone: {backbone}')

        config = backbone_configs[backbone]
        embed_dim = config['embed_dim']

        # Initialize CNN Stem
        stem_channels = []
        if use_cnn_stem:
            self.cnn_stem = CNNStem(in_channels, embed_dim)
            stem_channels = [embed_dim // 2]
        else:
            self.cnn_stem = None

        # Initialize Swin Transformer backbone
        self.backbone = self._create_backbone(
            backbone, in_channels, pretrained, pretrained_weights
        )

        # Calculate feature dimensions
        # Swin produces features at 4 scales with channel dimensions:
        # [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        backbone_channels = [embed_dim * (2**i) for i in range(4)]

        # Combine stem and backbone channels for decoder
        decoder_channels = stem_channels + backbone_channels

        # Initialize MDC Decoder
        self.decoder = MDCDecoder(
            in_channels=decoder_channels,
            num_classes=num_classes,
            dropout_ratio=decoder_dropout,
        )

    def _create_backbone(
        self,
        backbone: str,
        in_channels: int,
        pretrained: bool,
        pretrained_weights: str | None,
    ) -> nn.Module:
        """Create Swin Transformer backbone.

        Args:
            backbone: Model name ('swin_v2_t' or 'swin_v2_b').
            in_channels: Number of input channels.
            pretrained: Whether to use pretrained weights.
            pretrained_weights: Specific weight name for TorchGeo.

        Returns:
            Configured Swin Transformer model.
        """
        if pretrained_weights:
            # Use TorchGeo weights
            if backbone == 'swin_v2_t':
                weights_enum = Swin_V2_T_Weights
                model_fn = swin_v2_t
            else:
                weights_enum = Swin_V2_B_Weights
                model_fn = swin_v2_b

            # Get specific weights
            weights = getattr(weights_enum, pretrained_weights, None)
            if weights is None:
                raise ValueError(f'Unknown weights: {pretrained_weights}')

            model = model_fn(weights=weights)
        else:
            if backbone == 'swin_v2_t':
                model = swin_v2_t(weights='DEFAULT' if pretrained else None)
            else:
                model = swin_v2_b(weights='DEFAULT' if pretrained else None)

            # Adjust first layer for different input channels if needed
            if in_channels != 3:
                out_channels = model.features[0][0].out_channels
                model.features[0][0] = nn.Conv2d(
                    in_channels, out_channels, kernel_size=4, stride=4
                )

        # Wrap to extract multi-scale features
        return SwinBackboneWrapper(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AerialFormer.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        features = []

        # Extract CNN stem features
        if self.use_cnn_stem:
            stem_feat = self.cnn_stem(x)
            features.append(stem_feat)

        # Extract hierarchical Swin features
        backbone_features = self.backbone(x)
        features.extend(backbone_features)

        # Decode to segmentation map
        seg_logits = self.decoder(features)

        # Upsample to original resolution
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False
        )

        return seg_logits


class SwinBackboneWrapper(nn.Module):
    """Wrapper to extract multi-scale features from Swin Transformer.

    Args:
        swin_model: Pre-configured Swin Transformer model.
    """

    def __init__(self, swin_model: SwinTransformer) -> None:
        """Initialize Swin Backbone Wrapper.

        Args:
            swin_model: Pre-configured Swin Transformer model.
        """
        super().__init__()
        self.swin = swin_model

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract hierarchical features.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of feature tensors at different scales.
        """
        features = []

        # Process through Swin stages
        for i, stage in enumerate(self.swin.features):
            x = stage(x)
            # Extract features after each stage (excluding patch embedding)
            if i % 2 == 1:  # After downsample + blocks
                # Convert from (B, H, W, C) to (B, C, H, W)
                B, H, W, C = x.shape
                feat = x.permute(0, 3, 1, 2).contiguous()
                features.append(feat)

        return features


if __name__ == '__main__':
    """Test the AerialFormer implementation."""
    import time

    print('Testing AerialFormer implementation...')

    # Test configurations
    configs = [
        {
            'name': 'Tiny with CNN Stem',
            'backbone': 'swin_v2_t',
            'use_cnn_stem': True,
            'in_channels': 3,
        },
        {
            'name': 'Tiny without CNN Stem',
            'backbone': 'swin_v2_t',
            'use_cnn_stem': False,
            'in_channels': 3,
        },
        {
            'name': 'Base with CNN Stem',
            'backbone': 'swin_v2_b',
            'use_cnn_stem': True,
            'in_channels': 3,
        },
    ]

    # Add TorchGeo-specific test
    configs.append(
        {
            'name': 'Sentinel-2 Multi-spectral (TorchGeo)',
            'backbone': 'swin_v2_b',
            'use_cnn_stem': True,
            'in_channels': 9,
            'pretrained_weights': 'SENTINEL2_MI_MS_SATLAS',
        }
    )

    for config in configs:
        print(f'\n{"=" * 60}')
        print(f'Testing: {config["name"]}')
        print(f'{"=" * 60}')

        try:
            # Create model
            model = AerialFormer(
                num_classes=10,
                in_channels=config['in_channels'],
                backbone=config['backbone'],
                use_cnn_stem=config['use_cnn_stem'],
                pretrained_weights=config.get('pretrained_weights'),
            )
            model.eval()

            # Test input
            x = torch.randn(2, config['in_channels'], 512, 512)

            # Forward pass with timing
            with torch.no_grad():
                start_time = time.time()
                output = model(x)
                end_time = time.time()

            # Print results
            print('✓ Model created successfully')
            print(f'  Input shape:  {x.shape}')
            print(f'  Output shape: {output.shape}')
            print(f'  Forward time: {end_time - start_time:.3f} seconds')

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f'  Total params:     {total_params:,}')
            print(f'  Trainable params: {trainable_params:,}')

        except Exception as e:
            print(f'✗ Error: {e!s}')

    print(f'\n{"=" * 60}')
    print('Testing complete!')
