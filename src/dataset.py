import os
import glob
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset

# Fixed target shape for consistent tensor dimensions
TARGET_SHAPE = (160, 160, 160)

def pad_or_crop_to(x, target_shape):
    """
    Pad or crop a 3D volume to the target shape.

    Args:
        x (torch.Tensor or np.ndarray): Input volume [C, D, H, W].
        target_shape (tuple): Desired (D, H, W) dimensions.

    Returns:
        torch.Tensor: Volume padded/cropped to [C, tD, tH, tW].
    """
    # Convert numpy arrays to torch tensors
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    C, D, H, W = x.shape
    tD, tH, tW = target_shape

    # Compute necessary padding
    pad_d = max(0, tD - D)
    pad_h = max(0, tH - H)
    pad_w = max(0, tW - W)
    # F.pad expects (W_left, W_right, H_left, H_right, D_left, D_right)
    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    # Crop if input is larger than target
    x = x[:, :tD, :tH, :tW]
    return x


class NiftiDataset(Dataset):
    """
    PyTorch Dataset for 3D NIfTI images and labels from a TSV file.
    Automatically pads or crops volumes to TARGET_SHAPE.

    TSV must have columns: subject_id, session, diagnosis, age (optional).
    """
    def __init__(self, labels_path: str, images_dir: str, transform=None):
        # Load label table
        self.df = pd.read_csv(labels_path, sep='\t')
        self.images_dir = images_dir
        self.transform = transform

        # Build mapping: subject_id -> file path
        self.file_map = {}
        for ext in ('*.nii', '*.nii.gz'):
            pattern = os.path.join(images_dir, '**', ext)
            for fp in glob.glob(pattern, recursive=True):
                fname = os.path.basename(fp)
                tokens = fname.split('_')
                # Handle ADNI naming variation
                if tokens[0].upper() == 'ADNI' and len(tokens) >= 4:
                    sid = '_'.join(tokens[1:4])
                else:
                    sid = '_'.join(tokens[:3])
                self.file_map[sid] = fp

        # Keep only rows with available images
        self.df['subject_id'] = self.df['subject_id'].astype(str)
        self.df = self.df[self.df['subject_id'].isin(self.file_map)].reset_index(drop=True)

        # Map diagnoses to integers
        self.label_map = {'CN': 0, 'MCI': 1, 'AD': 2}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a sample with image tensor and label.

        Returns:
            sample (dict): {
                'image': torch.FloatTensor [1, D, H, W],
                'label': torch.LongTensor,  
                'subject_id': str,
                'age': float (optional)
            }
        """
        row = self.df.iloc[idx]
        sid = row['subject_id']
        diag = row['diagnosis']
        age = row.get('age', np.nan)

        # Load and expand dims to [C=1, D, H, W]
        nii = nib.load(self.file_map[sid])
        data = nii.get_fdata(dtype=np.float32)
        data = np.expand_dims(data, 0)

        # Build initial sample dict
        sample = {
            'image': data,
            'label': self.label_map.get(diag, -1),
            'subject_id': sid
        }
        if not np.isnan(age):
            sample['age'] = float(age)

        # Apply transforms (e.g., normalization, augmentation)
        if self.transform:
            sample = self.transform(sample)

        # Enforce fixed shape
        img = sample['image']
        img = pad_or_crop_to(img, TARGET_SHAPE)
        sample['image'] = img

        # Convert label to tensor
        lbl = sample['label']
        sample['label'] = torch.tensor(lbl, dtype=torch.long)

        return sample