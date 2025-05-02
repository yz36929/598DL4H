# src/train.py
import os
import yaml
import time
from torch.utils.tensorboard import SummaryWriter
import random
import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.dataset import NiftiDataset 
from src.model import AD3DCNN
from src.transforms import get_training_transforms


def load_config(path=None):
    """Load the configuration file."""
    if path is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the relative path to the config file
        path = os.path.join(script_dir, "..", "configs", "config.yaml")
    
    with open(path) as f:
     return yaml.safe_load(f)

def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    cfg = load_config()
    model.train()
    losses, preds, targs = [], [], []
    # for batch in loader:
    for batch_idx, batch in enumerate(loader):
        imgs, labels = batch['image'].to(device), batch['label'].to(device)
        if cfg['data']['include_age']:
            age = batch['age'].to(device)
            out = model(imgs, age)
        else:
            out = model(imgs)
        optimizer.zero_grad()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds += out.softmax(1).argmax(1).cpu().tolist()
        targs += labels.cpu().tolist()
        # ─── FINE‐GRAINED, PER‐BATCH TENSORBOARD LOGGING ───────────────────
        if hasattr(train_epoch, 'writer'):
            # compute a global_step = epoch*len(loader) + batch_idx + 1
            gs = train_epoch.current_epoch * len(loader) + batch_idx + 1
            train_epoch.writer.add_scalar('Loss/train_batch', loss.item(), gs)
            train_epoch.writer.add_scalar('Acc/train_batch', 
                                          (out.softmax(1).argmax(1)==labels).float().mean().item(),
                                          gs)
        # ────────────────────────────────────────────────────────────────────
    
    return (sum(losses)/len(losses),
            accuracy_score(targs, preds),
            balanced_accuracy_score(targs, preds))

def eval_epoch(model, loader, criterion, device):
    """Evaluate the model on the validation or test set."""
    cfg = load_config()
    model.eval()
    losses, preds, targs = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch['image'].to(device), batch['label'].to(device)
            if cfg['data']['include_age']:
                age = batch['age'].to(device)
                out = model(imgs, age)
            else:
                out = model(imgs)
            losses.append(criterion(out, labels).item())
            preds += out.softmax(1).argmax(1).cpu().tolist()
            targs += labels.cpu().tolist()
    return (sum(losses)/len(losses),
            accuracy_score(targs, preds),
            balanced_accuracy_score(targs, preds))

def main():
    """Main function to train and evaluate the model."""
    cfg = load_config()

    # 1) Make all our data paths absolute, relative to this file’s parent folder
    script_dir  = os.path.dirname(os.path.abspath(__file__))       # .../cnn_ad/src
    project_root = os.path.dirname(script_dir)                     # .../cnn_ad
    # prepend project_root to any relative paths in data block
    cfg['data']['preproc_dir'] = os.path.normpath(
        os.path.join(project_root,
                     cfg['data']['preproc_dir']))
    cfg['data']['labels']     = os.path.normpath(
        os.path.join(project_root,
                     cfg['data']['labels']))

    # sanity‐check what we’re about to load
    print(f"▶ Looking for NIfTIs in: {cfg['data']['preproc_dir']}")
    print(f"▶ Reading labels from:  {cfg['data']['labels']}")
    
    dev = torch.device(cfg['training']['device'])
    # ─── TENSORBOARD SETUP ─────────────────────────────────────────────────────
    # Create a SummaryWriter that will dump events into your configured log_dir
    writer = SummaryWriter(log_dir=cfg['paths']['log_dir'])
    # Log your hyperparameters once
    writer.add_hparams(
        {'lr': cfg['training']['lr'], 'batch_size': cfg['training']['batch_size'], 'epochs': cfg['training']['epochs']},
        {}
    )
    # ────────────────────────────────────────────────────────────────────────────

    # datasets
    # full_ds = NiftiDataset(cfg['data']['labels'], cfg['data']['preproc_dir'])
    # build two parallel datasets, one with aug, one raw
    raw_ds = NiftiDataset(
        cfg['data']['labels'],
        cfg['data']['preproc_dir'],
        transform=None
    )
    print(f"▶ raw_ds length: {len(raw_ds)}")
    aug_ds = NiftiDataset(
        cfg['data']['labels'],
        cfg['data']['preproc_dir'],
        transform=get_training_transforms()
    )
    print(f"▶ aug_ds length: {len(aug_ds)}")

    # split the dataset into train and validation sets
    total = len(raw_ds)
    n_train = int(total * cfg['data']['train_split'])
    n_val   = int(total * cfg['data']['val_split'])
    n_test = total - n_train - n_val

    # train_ds, val_ds, test_ds = torch.utils.data.random_split(
    #     full_ds, [n_train, n_val, n_test])
    if 'seed' in cfg['data']:
            random.seed(cfg['data']['seed'])
    indices = list(range(total))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train+n_val]
    test_idx  = indices[n_train+n_val:n_train+n_val+n_test]

    # create Subsets with augment only on train
    train_ds = Subset(aug_ds, train_idx)
    val_ds   = Subset(raw_ds, val_idx)
    test_ds  = Subset(raw_ds, test_idx)

    # dataloaders
    num_workers = cfg['training']['num_workers']
    train_loader = DataLoader(train_ds,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=False, num_workers=num_workers)

    # model, loss, opt
    model = AD3DCNN(in_channels=1,
                  num_classes=cfg['model']['num_classes'],
                  include_age=cfg['data']['include_age']).to(dev)
    crit = nn.CrossEntropyLoss()
    # opt  = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    # Ensure lr is a float
    lr = cfg['training']['lr']
    if isinstance(lr, str):
        lr = float(lr)
    opt  = optim.Adam(model.parameters(), lr=lr)

    best_bal = 0

    for ep in range(cfg['training']['epochs']):
        start_time = time.time() # mark start time

        # make writer & epoch visible to train_epoch
        train_epoch.writer = writer
        train_epoch.current_epoch = ep

        tr_loss, tr_acc, tr_bal = train_epoch(model, train_loader, crit, opt, dev)
        vl_loss, vl_acc, vl_bal = eval_epoch(model, val_loader, crit, dev)
  
        epoch_time = time.time() - start_time # compute elapsed time

        print(f"Epoch {ep+1:02d}: "
              f"Train {tr_loss:.3f}/{tr_acc:.3f}/{tr_bal:.3f} | "
              f"Val   {vl_loss:.3f}/{vl_acc:.3f}/{vl_bal:.3f}"
              f" | Time {epoch_time:.1f}s")
        # save the best model
        # ─── TENSORBOARD LOGGING ────────────────────────────────────────────────
        # scalars
        writer.add_scalar('Loss/train',    tr_loss, ep+1)
        writer.add_scalar('Loss/val',      vl_loss, ep+1)
        writer.add_scalar('Acc/train',     tr_acc, ep+1)
        writer.add_scalar('Acc/val',       vl_acc, ep+1)
        writer.add_scalar('BalAcc/train',  tr_bal, ep+1)
        writer.add_scalar('BalAcc/val',    vl_bal, ep+1)
        writer.add_scalar('Time/epoch(s)', epoch_time, ep+1)

        # optional: histogram of weights & biases
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, ep)

        # log computational graph once
        if ep == 0:
            sample_batch = next(iter(train_loader))
            imgs = sample_batch['image'].to(dev)
            if cfg['data']['include_age']:
                age = sample_batch['age'].to(dev)
                writer.add_graph(model, (imgs, age))
            else:
                writer.add_graph(model, imgs)
       # ────────────────────────────────────────────────────────────────────────────

        if vl_bal > best_bal:
            best_bal = vl_bal
            ckpt_path = os.path.join(cfg['paths']['ckpt_dir'],
                                      f"best_{vl_bal:.3f}.pt")
            torch.save(model.state_dict(), ckpt_path)

   # final test eval
    print(f"\nLoading best model({best_bal:.3f}) for test ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
    test_loss, test_acc, test_bal = eval_epoch(model, test_loader, crit, dev)
    print(f"▶ Test Loss: {test_loss:.3f} | "
          f"Test Acc: {test_acc:.3f} | "
          f"Test Bal: {test_bal:.3f}")
    # ─── CLOSE TENSORBOARD WRITER ─────────────────────────────────────────────
    writer.close()
    # ────────────────────────────────────────────────────────────────────────────

if __name__=="__main__":
    main()
