import os
import urllib.request

import numpy as np
import requests
import anndata as ad
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import FIGSHARE_ARTICLE, PANCREAS_FILE, N_TOP_GENES, TRAIN_FRACTION, BATCH_SIZE

sc.settings.verbosity = 0

_STUDY_NAMES = {
    'smarter':   'Xin 2016',
    'smartseq2': 'Segerstolpe 2016',
    'celseq2':   'Muraro 2016',
    'inDrop1':   'Baron 2016 (batch 1)',
    'inDrop2':   'Baron 2016 (batch 2)',
    'inDrop3':   'Baron 2016 (batch 3)',
    'inDrop4':   'Baron 2016 (batch 4)',
}

_ALL_TECHS = set(_STUDY_NAMES.keys())


def download_pancreas(save_path: str = PANCREAS_FILE) -> ad.AnnData:
    """
    Download the Human Pancreas scib benchmark h5ad from figshare (~301 MB).
    Cached after first run.

    Source: Luecken et al. Nature Methods 2022, figshare article 12420968.
    16,382 cells × 19,093 genes | 14 cell types | 4 sequencing technologies.
    Data is pre-normalised — skip normalize_total / log1p.
    """
    if os.path.exists(save_path):
        print(f'Cache found: {save_path}')
        return ad.read_h5ad(save_path)

    print('Querying figshare API...')
    resp  = requests.get(f'https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}/files')
    files = resp.json()

    target = next((f for f in files if 'pancreas' in f['name'].lower()), None)
    if target is None:
        raise FileNotFoundError(
            f'No pancreas file in figshare article {FIGSHARE_ARTICLE}. '
            f'Available: {[f["name"] for f in files]}'
        )

    size_mb = target['size'] // 1024 // 1024
    print(f"Downloading '{target['name']}' ({size_mb} MB)...")

    def _progress(count, block, total):
        if count % 300 == 0:
            print(f'  {min(count * block * 100 / total, 100):.0f}%', end='\r')

    urllib.request.urlretrieve(target['download_url'], save_path, _progress)
    print('\nDownload complete.')
    return ad.read_h5ad(save_path)


def prepare_pancreas(adata: ad.AnnData, n_top_genes: int = N_TOP_GENES,
                     test_tech: str = 'smarter'):
    """
    Cross-technology split for the Human Pancreas scib benchmark.
    Data is pre-normalised — only HVG selection is applied here.

    Protocol → study mapping (Luecken et al. 2022):
        inDrop1-4  → Baron 2016
        celseq2    → Muraro 2016
        smartseq2  → Segerstolpe 2016
        smarter    → Xin 2016

    Protocols not in the original 4-study benchmark (celseq=Grün,
    fluidigmc1=Lawlor) are excluded.

    Parameters
    ----------
    test_tech : str
        Protocol to hold out as test. Default 'smarter' (Xin, 4 islet types).
        Use 'smartseq2' (Segerstolpe) for a harder 13-type evaluation.
    """
    if test_tech not in _ALL_TECHS:
        raise ValueError(
            f"Unknown test_tech '{test_tech}'. Choose from: {sorted(_ALL_TECHS)}"
        )

    label_col   = 'celltype'
    tech_col    = 'tech'
    train_techs = sorted(_ALL_TECHS - {test_tech})

    print(f'Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes')

    adata = adata[adata.obs[tech_col].isin(_ALL_TECHS)].copy()
    print(f'After filtering to original 4 studies: {adata.shape[0]} cells')

    print(f'Cell types ({adata.obs[label_col].nunique()}):')
    print(adata.obs[label_col].value_counts().to_string())

    print(f'\nTrain : {train_techs}')
    print(f'Test  : {test_tech}  ({_STUDY_NAMES[test_tech]})')

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True, flavor='seurat')
    print(f'After HVG selection: {adata.shape[1]} genes')

    mask        = adata.obs[tech_col].isin(train_techs)
    train_adata = adata[mask].copy()
    test_adata  = adata[~mask].copy()

    n_test_types = test_adata.obs[label_col].nunique()
    print(f'Train: {train_adata.shape[0]} cells | Test: {test_adata.shape[0]} cells '
          f'({n_test_types} cell types in test)')

    return train_adata, test_adata, label_col


def _to_array(adata: ad.AnnData) -> np.ndarray:
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return X.astype(np.float32)


def build_loaders(train_adata: ad.AnnData, test_adata: ad.AnnData,
                  label_col: str, seed: int,
                  train_fraction: float = TRAIN_FRACTION,
                  batch_size: int = BATCH_SIZE):
    """
    Build train / val / test DataLoaders.

    - LabelEncoder is fit on training cell types only.
    - Val split (15%) is taken before stratified subsampling to avoid leakage.
    - Training uses a stratified subsample (default 10%) for rare-type stress testing.
    - Test cells with unseen cell types are dropped.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    le    = LabelEncoder().fit(train_adata.obs[label_col].values)
    X_all = _to_array(train_adata)
    y_all = le.transform(train_adata.obs[label_col].values)

    idx   = np.random.permutation(len(y_all))
    val_n = int(len(y_all) * 0.15)
    X_val, y_val     = X_all[idx[:val_n]], y_all[idx[:val_n]]
    X_tr_full, y_tr_full = X_all[idx[val_n:]], y_all[idx[val_n:]]

    # Drop cell types with fewer than 2 samples before stratified split
    ct_counts = np.bincount(y_tr_full, minlength=len(le.classes_))
    valid_cls = np.where(ct_counts >= 2)[0]
    mask      = np.isin(y_tr_full, valid_cls)
    Xf, yf   = X_tr_full[mask], y_tr_full[mask]
    keep, _  = train_test_split(
        np.arange(len(yf)), train_size=train_fraction,
        stratify=yf, random_state=seed
    )
    X_tr, y_tr = Xf[keep], yf[keep]

    test_labels = test_adata.obs[label_col].values
    test_mask   = np.isin(test_labels, le.classes_)
    X_te = _to_array(test_adata)[test_mask]
    y_te = le.transform(test_labels[test_mask])

    def _loader(X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        _loader(X_tr, y_tr, shuffle=True),
        _loader(X_val, y_val),
        _loader(X_te, y_te),
        le, X_tr.shape[1], len(le.classes_),
    )
