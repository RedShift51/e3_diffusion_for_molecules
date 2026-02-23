"""
QM9 via PyTorch Geometric. Loads from torch_geometric.datasets.QM9 and converts
to the dict format expected by PreprocessQM9 (positions, one_hot, charges).

When RDKit is installed, the default QM9() tries to process raw SDF and can hit
None from the parser (AttributeError: 'NoneType' object has no attribute 'GetNumAtoms').
We use QM9Preprocessed to always load the pre-processed .pt from PyG, avoiding RDKit.
"""
import logging
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Subset

# Atomic number -> index for one_hot (H=0, C=1, N=2, O=3, F=4)
Z_TO_IDX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
N_ATOM_TYPES = 5

log = logging.getLogger(__name__)

# Pre-processed QM9 URL (same as PyG when RDKit is not installed)
QM9_PROCESSED_URL = 'https://data.pyg.org/datasets/qm9_v3.zip'


class QM9Preprocessed(torch.utils.data.Dataset):
    """
    QM9 from PyG's pre-processed file (qm9_v3.pt) only. Avoids RDKit parsing of raw SDF,
    which can fail for some molecules and raise AttributeError on mol.GetNumAtoms().
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        from torch_geometric.data import InMemoryDataset, download_url, extract_zip
        from torch_geometric.io import fs
        from torch_geometric.data import Data

        self.root = osp.expanduser(root)
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_path = osp.join(self.processed_dir, 'data_v3.pt')
        self.raw_pt = osp.join(self.raw_dir, 'qm9_v3.pt')

        if not osp.isfile(self.processed_path) or force_reload:
            raw_pt = self.raw_pt
            if not osp.isfile(raw_pt) or force_reload:
                log.info('Downloading pre-processed QM9 from %s', QM9_PROCESSED_URL)
                path = download_url(QM9_PROCESSED_URL, self.raw_dir)
                extract_zip(path, self.raw_dir)
                if osp.isfile(path):
                    os.unlink(path)
                # Locate .pt file (zip may put it in raw_dir or raw_dir/qm9/)
                for candidate in [osp.join(self.raw_dir, 'qm9_v3.pt'), osp.join(self.raw_dir, 'qm9', 'qm9_v3.pt')]:
                    if osp.isfile(candidate):
                        raw_pt = candidate
                        break
            log.info('Loading pre-processed QM9 into memory...')
            data_list = fs.torch_load(raw_pt)
            if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
                data_list = [Data(**d) for d in data_list]
            if pre_filter is not None:
                data_list = [d for d in data_list if pre_filter(d)]
            if pre_transform is not None:
                data_list = [pre_transform(d) for d in data_list]
            torch.save(data_list, self.processed_path)

        self._data_list = torch.load(self.processed_path, weights_only=False)
        self.transform = transform

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        data = self._data_list[idx]
        if isinstance(data, dict):
            data = Data(**data)
        if self.transform is not None:
            data = self.transform(data)
        return data


def _pyg_data_to_mol_dict(data, load_charges=True):
    """Convert one PyG Data to dict with positions, one_hot, charges."""
    pos = data.pos  # (n_atoms, 3)
    z = data.z      # (n_atoms,) atomic numbers 1,6,7,8,9
    n = pos.size(0)
    # one_hot: (n_atoms, 5)
    idx = torch.zeros(n, dtype=torch.long, device=z.device)
    for i, atomic in enumerate([1, 6, 7, 8, 9]):
        idx[z == atomic] = i
    one_hot = torch.nn.functional.one_hot(idx, num_classes=N_ATOM_TYPES).float()
    # charges: mask (1 for real atom; no partial charges from PyG)
    charges = torch.ones(n, dtype=torch.float, device=pos.device)
    return {
        'positions': pos,
        'one_hot': one_hot,
        'charges': charges,
    }


class QM9PyGDataset(torch.utils.data.Dataset):
    """Wrapper that yields per-molecule dicts from PyG QM9 for the existing collate."""
    def __init__(self, pyg_dataset, load_charges=True):
        self.pyg_dataset = pyg_dataset
        self.load_charges = load_charges

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, i):
        data = self.pyg_dataset[i]
        return _pyg_data_to_mol_dict(data, load_charges=self.load_charges)


def get_pyg_qm9_splits(root='./data/qm9', train_size=100_000, test_ratio=0.1, seed=0):
    """
    Load QM9 via pre-processed PyG data (no RDKit parsing) and split into train/valid/test.
    Returns indices dict and full dataset. Uses QM9Preprocessed to avoid RDKit None-mol errors.
    """
    log.info('Loading QM9 from PyTorch Geometric pre-processed (root=%s)', root)
    full = QM9Preprocessed(root=root)
    log.info('Loaded %d molecules', len(full))
    n = len(full)
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_test = int(n * test_ratio)
    n_valid = n_test
    n_train = min(train_size, n - n_test - n_valid)
    # train: first n_train, valid: next n_valid, test: next n_test
    train_idx = perm[:n_train]
    valid_idx = perm[n_train:n_train + n_valid]
    test_idx = perm[n_train + n_valid:n_train + n_valid + n_test]
    return {
        'train': train_idx,
        'valid': valid_idx,
        'test': test_idx,
    }, full


def retrieve_dataloaders_pyg_qm9(cfg):
    """Build train/valid/test DataLoaders from PyG QM9."""
    from torch.utils.data import DataLoader
    from qm9.data.collate import PreprocessQM9

    root = getattr(cfg, 'qm9_root', './data/qm9')
    splits, full_dataset = get_pyg_qm9_splits(
        root=root,
        train_size=getattr(cfg, 'qm9_train_size', 100_000),
        test_ratio=0.1,
        seed=42,
    )
    wrapper = QM9PyGDataset(full_dataset, load_charges=cfg.include_charges)
    batch_size = cfg.batch_size
    num_workers = getattr(cfg, 'num_workers', 0)
    prefetch_factor = getattr(cfg, 'prefetch_factor', 2)
    preprocess = PreprocessQM9(load_charges=cfg.include_charges)

    dataloaders = {}
    for split_name, indices in splits.items():
        subset = Subset(wrapper, indices)
        dl_kw = dict(
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            collate_fn=preprocess.collate_fn,
        )
        if num_workers > 0:
            dl_kw['prefetch_factor'] = prefetch_factor
        dataloaders[split_name] = DataLoader(subset, **dl_kw)

    # valid key: main_qm9 uses dataloaders['valid'], old code used 'valid'
    if 'valid' not in dataloaders and 'val' in dataloaders:
        dataloaders['valid'] = dataloaders['val']
    return dataloaders, None
