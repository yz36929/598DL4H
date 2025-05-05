"""
Training script for 3D CNN on ADNI NIfTI volumes with optional age incorporation,
weighted sampling, padding/cropping collate and TensorBoard logging.
"""
import os
import yaml
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from src.dataset import NiftiDataset
from src.model import AD3DCNN
from src.transforms import (
    get_training_transforms,
    Normalize,
    ToTensor,
    RandomFlip,
    Rotate3D,
    Translate3D,
    Compose
)

def load_config(path: str = None) -> dict:
    """
    Load and return the YAML configuration.

    Args:
        path: Optional custom path to config file.
    Returns:
        A dictionary of configuration parameters.
    """
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "..", "configs", "config.yaml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def pad_or_crop_to(x: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Pad or crop a single-volume tensor to a target shape.

    Args:
        x: Tensor of shape [C, D, H, W].
        target_shape: Desired (D, H, W) shape.
    Returns:
        Padded or cropped tensor of shape [C, tD, tH, tW].
    """
    C, D, H, W = x.shape
    tD, tH, tW = target_shape

    # Compute padding widths
    pad_d = max(0, tD - D)
    pad_h = max(0, tH - H)
    pad_w = max(0, tW - W)
    # Apply padding: (W_left, W_right, H_left, H_right, D_left, D_right)
    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    # Crop if larger than target
    return x[:, :tD, :tH, :tW]

def pad_collate(batch: list) -> dict:
    """
    Collate function for DataLoader to pad/crop variable-size volumes in a batch.

    Args:
        batch: List of samples, each a dict with 'image', 'label', optionally 'age'.
    Returns:
        A dict with batched 'image', 'label', and optionally 'age'.
    """
    images = [b['image'] for b in batch]
    labels = torch.stack([b['label'] for b in batch], dim=0).long()
    ages = None
    if 'age' in batch[0]:
        ages = torch.tensor([b['age'] for b in batch], dtype=torch.float32)

    # Determine max shape in batch
    Ds, Hs, Ws = zip(*(img.shape[-3:] for img in images))
    target = (max(Ds), max(Hs), max(Ws))

    # Pad/crop each image then stack
    padded = [pad_or_crop_to(img, target) for img in images]
    images = torch.stack(padded, dim=0).float()

    result = {'image': images, 'label': labels}
    if ages is not None:
        result['age'] = ages
    return result

def train_epoch(model, loader, criterion, optimizer, device) -> tuple:
    """
    Run one training epoch.

    Args:
        model: The 3D CNN model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Torch device.
    Returns:
        Tuple of (loss, accuracy, balanced_accuracy).
    """
    cfg = load_config()
    model.train()
    losses, preds, targets = [], [], []
    use_age = cfg['data']['include_age']

    for batch_idx, batch in enumerate(loader):
        imgs = batch['image'].float().to(device)
        labels = batch['label'].long().to(device)

        # Forward pass with or without age
        if use_age:
            age = batch['age'].float().to(device)
            outputs = model(imgs, age)
        else:
            outputs = model(imgs)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(outputs.softmax(1).argmax(1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

        # Per-batch TensorBoard logging
        if hasattr(train_epoch, 'writer'):
            gs = train_epoch.current_epoch * len(loader) + batch_idx + 1
            train_epoch.writer.add_scalar('Loss/train_batch', loss.item(), gs)
            acc_batch = (outputs.softmax(1).argmax(1) == labels).float().mean().item()
            train_epoch.writer.add_scalar('Acc/train_batch', acc_batch, gs)

    avg_loss = sum(losses) / len(losses)
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    return avg_loss, acc, bal_acc

def eval_epoch(model, loader, criterion, device) -> tuple:
    """
    Run one evaluation epoch (validation or test).

    Args:
        model: The 3D CNN model.
        loader: DataLoader for validation/test.
        criterion: Loss function.
        device: Torch device.
    Returns:
        Tuple of (loss, accuracy, balanced_accuracy, auc_roc).
    """
    cfg = load_config()
    model.eval()
    losses, preds, targets = [], [], []
    use_age = cfg['data']['include_age']
    all_probs = []   # to store softmax probabilities

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            if use_age:
                age = batch['age'].to(device)
                outputs = model(imgs, age)
            else:
                outputs = model(imgs)

            losses.append(criterion(outputs, labels).item())
            preds.extend(outputs.softmax(1).argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
            all_probs.extend(outputs.softmax(1).cpu().tolist())

    avg_loss = sum(losses) / len(losses)
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)

    # one‐hot encode the true labels
    y_true  = label_binarize(targets, classes=[0,1,2])
    y_score = np.array(all_probs)   # shape [N,3]

    auc_roc = roc_auc_score(
        y_true,
        y_score,
        average='macro',
        multi_class='ovo'
    )
    return avg_loss, acc, bal_acc, auc_roc

def main():
    """
    Main training and evaluation loop.
    """
    cfg = load_config()

    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    cfg['data']['preproc_dir'] = os.path.normpath(
        os.path.join(project_root, cfg['data']['preproc_dir']))
    cfg['data']['labels'] = os.path.normpath(
        os.path.join(project_root, cfg['data']['labels']))

    print(f"▶ NIfTIs in: {cfg['data']['preproc_dir']}")
    print(f"▶ Labels from: {cfg['data']['labels']}")

    device = torch.device(cfg['training']['device'])

    # TensorBoard setup
    writer = SummaryWriter(log_dir=cfg['paths']['log_dir'])
    writer.add_hparams(
        {'lr': cfg['training']['lr'],
         'batch_size': cfg['training']['batch_size'],
         'epochs': cfg['training']['epochs']},
        {}
    )

    # Prepare datasets and transforms
    val_transforms = Compose([Normalize(), ToTensor()])
    train_transforms = Compose([Normalize(), ToTensor()])

    raw_ds = NiftiDataset(
        cfg['data']['labels'], cfg['data']['preproc_dir'], transform=val_transforms)
    aug_ds = NiftiDataset(
        cfg['data']['labels'], cfg['data']['preproc_dir'], transform=train_transforms)

    total = len(raw_ds)
    n_train = int(total * cfg['data']['train_split'])
    n_val = int(total * cfg['data']['val_split'])
    n_test = total - n_train - n_val

    if 'seed' in cfg['data']:
        random.seed(cfg['data']['seed'])
    indices = list(range(total))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:n_train+n_val+n_test]

    train_ds = Subset(aug_ds, train_idx)
    val_ds = Subset(raw_ds, val_idx)
    test_ds = Subset(raw_ds, test_idx)

    # Weighted sampling on training split
    labels_arr = raw_ds.df['diagnosis'].map({'CN': 0, 'MCI': 1, 'AD': 2}).values
    counts = np.bincount(labels_arr)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels_arr]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights[train_idx],
        num_samples=len(train_idx),
        replacement=True
    )

    # DataLoaders
    num_workers = cfg['training']['num_workers']
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate
    )

    # Model, loss, optimizer, scheduler
    model = AD3DCNN(
        in_channels=1,
        num_classes=cfg['model']['num_classes'],
        include_age=cfg['data']['include_age']
    ).to(device)

    # # ev impl3 - Make the network wider
    # model = AD3DCNN(
    #     in_channels=1,
    #     num_classes=cfg['model']['num_classes'],
    #     base_filters=64,                     # ← doubled from 32
    #     include_age=cfg['data']['include_age']
    # ).to(device)

    weights = torch.tensor(1.0 / counts, device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    lr_val = float(cfg['training']['lr'])
    optimizer = optim.Adam(model.parameters(), lr=lr_val, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_bal = 0.0
    for ep in range(cfg['training']['epochs']):
        start = time.time()
        train_epoch.writer = writer
        train_epoch.current_epoch = ep

        tr_loss, tr_acc, tr_bal = train_epoch(
            model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_bal,vl_auc = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        elapsed = time.time() - start
        print(f"Epoch {ep+1:02d}: "
        f"Train {tr_loss:.3f}/{tr_acc:.3f}/{tr_bal:.3f} | "
        f"Val   {vl_loss:.3f}/{vl_acc:.3f}/{vl_bal:.3f}/{vl_auc:.3f} | "
        f"Time {elapsed:.1f}s")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', tr_loss, ep+1)
        writer.add_scalar('Loss/val', vl_loss, ep+1)
        writer.add_scalar('Acc/train', tr_acc, ep+1)
        writer.add_scalar('Acc/val', vl_acc, ep+1)
        writer.add_scalar('BalAcc/train', tr_bal, ep+1)
        writer.add_scalar('BalAcc/val', vl_bal, ep+1)
        writer.add_scalar('AUC/val', vl_auc, ep+1)
        writer.add_scalar('Time/epoch', elapsed, ep+1)

        if vl_bal > best_bal:
            best_bal = vl_bal
            ckpt_path = os.path.join(
                cfg['paths']['ckpt_dir'], f"best_{best_bal:.3f}.pt")
            torch.save(model.state_dict(), ckpt_path)

    # Final test evaluation
    print(f"\nLoading best model ({best_bal:.3f}) for test...")
    # only load the raw tensor weights (no pickle) to avoid FutureWarning
    model.load_state_dict(
        torch.load(ckpt_path,
                   map_location=device,
                   weights_only=True)
    )
    te_loss, te_acc, te_bal, te_auc = eval_epoch(model, test_loader, criterion, device)
    print(f"▶ Test Loss: {te_loss:.3f} | Test Acc: {te_acc:.3f} | Test Bal: {te_bal:.3f} | Test AUC: {te_auc:.3f}")

    writer.close()


if __name__ == '__main__':
    main()
