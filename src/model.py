import torch
import torch.nn as nn
import torch.nn.functional as F

class AD3DCNN(nn.Module):
    """
    A 3D convolutional network for Alzheimer's classification,
    inspired by [Hao et al., 2019] and aligned with the target paper.
    Architecture: 3 conv blocks + adaptive pooling + 2 FC layers.
    """
    def __init__(self, in_channels=1, num_classes=3,
                  base_filters=32, dropout_p=0.5,
                  include_age=False):
        super(AD3DCNN, self).__init__()
        self.include_age = include_age
        # Conv Block 1: in_channels -> base_filters
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        # Conv Block 2: base_filters -> base_filters*2
        self.block2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        # Conv Block 3: base_filters*2 -> base_filters*4
        self.block3 = nn.Sequential(
            nn.Conv3d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        # Adaptive pooling to fixed-size feature map
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        # Fully connected layers
        fc_input_dim = base_filters*4 * 4 * 4 * 4
        if include_age:
            fc_input_dim += 1
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x,age=None):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
            age (torch.Tensor, optional): Age tensor of shape [B]. Required if include_age is True.
        Returns:
            torch.Tensor: Output logits of shape [B, num_classes].
        """
        # x shape: [B, C=1, D, H, W]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        # flatten
        x = x.view(x.size(0), -1)
        if self.include_age:
            if age is None:
                raise ValueError("Age must be provided when include_age is True.")
            a = age.view(-1,1).to(x.dtype)
            x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

if __name__ == "__main__":
    # quick shape check
    model = AD3DCNN(in_channels=1, num_classes=3)
    dummy = torch.randn(2,1,64,64,64)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2,3]
