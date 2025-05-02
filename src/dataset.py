# src/dataset.py
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib

class NiftiDataset(Dataset):
    """
    PyTorch Dataset for loading 3D NIfTI images and labels from a TSV file.
    Supports optional transforms applied on a sample dict.
    """
    def __init__(self, labels_path, images_dir, transform=None):
        # Load the labels table
        self.df = pd.read_csv(labels_path, sep='\t')
        self.images_dir = images_dir
        self.transform = transform

        # Build a mapping from subject_id to image file path
        self.file_map = {}
        for ext in ('*.nii', '*.nii.gz'):
            pattern = os.path.join(images_dir,'**', ext)
            for fp in glob.glob(pattern, recursive=True):
                name = os.path.basename(fp)
                # assume subject_id is first three tokens: e.g. '016_S_0769'
                tokens = name.split('_')
                if tokens[0].upper() == 'ADNI' and len(tokens) >= 4:
                    sid = '_'.join(tokens[1:4])
                else:
                    sid = '_'.join(tokens[:3])
                self.file_map[sid] = fp

        # Filter label rows to those with an image file
        self.df['subject_id'] = self.df['subject_id'].astype(str)
        self.df = self.df[self.df['subject_id'].isin(self.file_map)].reset_index(drop=True)

        # Label to index mapping
        self.label_map = {'CN': 0, 'MCI': 1, 'AD': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row['subject_id']
        diag = row['diagnosis']
        age = row.get('age', np.nan)

        # Load image as numpy array
        nii = nib.load(self.file_map[sid])
        data = nii.get_fdata(dtype=np.float32)
        data = np.expand_dims(data, 0)  # → [C=1, D, H, W]

        # Build sample dict
        sample = {
            'image': data,
            'label': self.label_map.get(diag, -1),
            'subject_id': sid
        }
        if not np.isnan(age):
            sample['age'] = float(age)

        # Apply transforms which expect a dict
        if self.transform:
            sample = self.transform(sample)

        # Ensure tensors
        img = sample['image']
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).float()
        sample['image'] = img

        lbl = sample['label']
        if not torch.is_tensor(lbl):
            sample['label'] = torch.tensor(lbl, dtype=torch.long)

        return sample