import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class Compose:
    """
    Chains multiple transforms and applies them sequentially to a sample.

    Args:
        transforms (List[callable]): List of transform functions/classes.
    """
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    """
    Converts the 'image' field in the sample from a NumPy array to a PyTorch tensor.

    Expects:
        sample['image']: numpy.ndarray of shape [C, D, H, W]
    Returns:
        sample with 'image' as torch.FloatTensor
    """
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = sample['image']
        sample['image'] = torch.from_numpy(img).float()
        return sample


class Normalize:
    """
    Z-score normalization of the image: (x - mean) / std.

    Args:
        mean (Optional[float]): Precomputed mean. If None, compute from data.
        std (Optional[float]): Precomputed std. If None, compute from data.
    """
    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = sample['image']
        m = self.mean if self.mean is not None else float(np.mean(img))
        s = self.std if self.std is not None else float(np.std(img))
        sample['image'] = (img - m) / (s + 1e-8)
        return sample


class RandomFlip:
    """
    Randomly flips the 3D image along specified spatial axes.

    Args:
        axes (Tuple[int, ...]): Axes to consider for flipping (0=D,1=H,2=W).
    """
    def __init__(self, axes: Tuple[int, ...] = (0, 1, 2)):
        self.axes = axes

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = sample['image']
        for ax in self.axes:
            if random.random() > 0.5:
                # +1 offset because channel dimension is 0
                img = np.flip(img, axis=ax + 1)
        sample['image'] = img.copy()
        return sample


class Rotate3D:
    """
    Randomly rotates the image by multiples of 90° around a random axis pair.

    Args:
        axes (Tuple[Tuple[int, int], ...]): Pairs of spatial axes to rotate around.
        prob (float): Probability of applying the rotation.
    """
    def __init__(self,
                 axes: Tuple[Tuple[int, int], ...] = ((1, 2), (1, 3), (2, 3)),
                 prob: float = 0.5):
        self.axes = axes
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob:
            img = sample['image']
            axis_pair = random.choice(self.axes)
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k=k, axes=(axis_pair[0] + 1, axis_pair[1] + 1)).copy()
            sample['image'] = img
        return sample


class Translate3D:
    """
    Randomly translates the image by up to max_shift voxels in each spatial dimension.

    Args:
        max_shift (int): Maximum shift in voxels.
        prob (float): Probability of applying the translation.
    """
    def __init__(self, max_shift: int = 5, prob: float = 0.5):
        self.max_shift = max_shift
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob:
            img = sample['image']
            shifts = [random.randint(-self.max_shift, self.max_shift) for _ in range(3)]
            pad_width = [(0, 0)] + [(self.max_shift, self.max_shift)] * 3
            padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
            dz, dy, dx = shifts
            C, Dp, Hp, Wp = padded.shape
            d0 = self.max_shift + dz
            h0 = self.max_shift + dy
            w0 = self.max_shift + dx
            img = padded[:,
                         d0:d0 + img.shape[1],
                         h0:h0 + img.shape[2],
                         w0:w0 + img.shape[3]].copy()
            sample['image'] = img
        return sample


def get_training_transforms() -> Compose:
    """
    Returns a standard pipeline of augmentations for training:
      - Z-score normalization
      - Random flips
      - Random 90° rotations
      - Random translations
      - Conversion to tensor
    """
    return Compose([
        Normalize(),
        RandomFlip(),
        Rotate3D(prob=0.7),
        Translate3D(prob=0.7),
        ToTensor(),
    ])