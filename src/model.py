import torch
import torch.nn as nn
import torch.nn.functional as F

class AD3DCNN(nn.Module):
    """
    A 3D convolutional network for Alzheimer's classification.
    Inspired by Hao et al. (2019) with 3 convolutional blocks, adaptive pooling,
    and two fully-connected layers. Optionally incorporates age as a feature.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for T1).
        num_classes (int): Number of output classes (e.g., 3 for CN/MCI/AD).
        base_filters (int): Number of filters in the first conv block.
        dropout_p (float): Dropout probability after first FC.
        include_age (bool): Whether to append age to feature vector.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_filters: int = 32,
        dropout_p: float = 0.3,
        include_age: bool = False,
    ):
        super().__init__()
        self.include_age = include_age

        # Convolutional Block 1: [in] -> [base_filters]
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_filters),
            # nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),  # downsample by 2
            # nn.MaxPool3d(kernel_size=4),  # downsample by 4
        )

        # Convolutional Block 2: [base_filters] -> [base_filters * 2]
        self.block2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_filters * 2),
            # nn.BatchNorm3d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )

        # Convolutional Block 3: [base_filters*2] -> [base_filters*4]
        self.block3 = nn.Sequential(
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_filters * 4),
            # nn.BatchNorm3d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )
        # # ev impl3 - add a 4th conv block to make the network deeper
        # # — NEW: Conv Block 4: [base_filters*4] -> [base_filters*8]
        # self.block4 = nn.Sequential(
        #     nn.Conv3d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
        #     nn.InstanceNorm3d(base_filters * 8),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=2),
        # )

        # Adaptive average pooling to fixed spatial size (4x4x4)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        # Compute input dimension for FC layer
        fc_in_dim = base_filters * 4 * 4 * 4 * 4
        # # ev impl3 - add a 4th conv block to make the network deeper
        # # Now after 4 blocks, channels = base_filters * 8
        # fc_in_dim = base_filters * 8 * 4 * 4 * 4
        if include_age:
            fc_in_dim += 1  # extra age feature

        # Fully-connected layers
        self.fc1 = nn.Linear(fc_in_dim, 256)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, age: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input volume of shape [B, C, D, H, W].
            age (torch.Tensor, optional): Age feature of shape [B].

        Returns:
            torch.Tensor: Logits of shape [B, num_classes].
        """
        # Pass through conv blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # # ev impl3 - add a 4th conv block to make the network deeper
        # x = self.block4(x)   # new depth

        # Pool and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Append age if requested
        if self.include_age:
            if age is None:
                raise ValueError("Age must be provided when include_age is True.")
            age_feat = age.view(-1, 1).to(x.dtype)
            x = torch.cat([x, age_feat], dim=1)

        # Fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


if __name__ == "__main__":
    # Quick shape check with dummy input
    model = AD3DCNN(in_channels=1, num_classes=3)
    dummy = torch.randn(2, 1, 64, 64, 64)
    output = model(dummy)
    print("Output shape:", output.shape)  # Expected: [2, 3]
