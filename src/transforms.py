# src/transforms.py
import random
import numpy as np
import torch
import nibabel as nib


class Compose:
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    """
    Convert numpy array to PyTorch Tensor.
    Expects sample to be {'image': numpy.ndarray, ...}
    """
    def __call__(self, sample):
        img = sample['image']
        tensor = torch.from_numpy(img).float()
        sample['image'] = tensor
        return sample


class Normalize:
    """
    Z-score normalization: (x - mean) / std
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        if self.mean is None:
            mean = img.mean()
        else:
            mean = self.mean
        if self.std is None:
            std = img.std()
        else:
            std = self.std
        sample['image'] = (img - mean) / (std + 1e-8)
        return sample


class RandomFlip:
    """
    Randomly flip along given axes.
    """
    def __init__(self, axes=(0, 1, 2)):
        self.axes = axes

    def __call__(self, sample):
        img = sample['image']
        for ax in self.axes:
            if random.random() > 0.5:
                img = np.flip(img, axis=ax + 1)  # +1 because channel dim is 0
        sample['image'] = img.copy()
        return sample


class Rotate3D:
    """
    Random rotation by multiples of 90 degrees around a random axis.
    """
    def __init__(self, axes=((1,2),(1,3),(2,3)), prob=0.5):
        self.axes = axes
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            img = sample['image']
            axis = random.choice(self.axes)
            k = random.choice([1,2,3])  # number of 90 deg rotations
            # Rotate on spatial dims (img shape [C,D,H,W])
            img = np.rot90(img, k=k, axes=axis).copy()
            sample['image'] = img
        return sample


class Translate3D:
    """
    Random translation by a few pixels in each dimension.
    """
    def __init__(self, max_shift=5, prob=0.5):
        self.max_shift = max_shift
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            img = sample['image']
            shifts = [random.randint(-self.max_shift, self.max_shift) for _ in range(3)]
            # pad and shift
            pad = [(0,0)] + [(self.max_shift, self.max_shift)]*3
            img_padded = np.pad(img, pad_width=pad, mode='constant', constant_values=0)
            d0, d1, d2 = shifts
            C, Dp, Hp, Wp = img_padded.shape
            # crop back
            d_start = self.max_shift + d0
            h_start = self.max_shift + d1
            w_start = self.max_shift + d2
            img = img_padded[
                :,
                d_start:d_start+img.shape[1],
                h_start:h_start+img.shape[2],
                w_start:w_start+img.shape[3]
            ].copy()
            sample['image'] = img
        return sample


def get_training_transforms():
    """
    Return a Compose of data augmentation transforms.
    """
    return Compose([
        Normalize(),
        RandomFlip(),
        Rotate3D(prob=0.7),        
        Translate3D(prob=0.7),     
        ToTensor(),
    ])
