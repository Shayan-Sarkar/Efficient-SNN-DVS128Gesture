from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .fast_parser import load_polarity_events
from .representation import events_to_voxel_grid


def _read_trials(root: Path, split: str) -> list[str]:
    fn = 'trials_to_train.txt' if split == 'train' else 'trials_to_test.txt'
    return [l.strip() for l in (root / fn).read_text().splitlines() if l.strip()]


def _read_labels(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                'label': int(row['class']) - 1,
                'start': int(row['startTime_usec']),
                'end':   int(row['endTime_usec']),
            })
    return rows


def build_cache(
    dataset_root: str | Path,
    cache_dir: str | Path,
    T: int = 20,
    H: int = 64,
    W: int = 64,
    force: bool = False,
) -> None:
    from tqdm import tqdm

    dataset_root = Path(dataset_root)
    cache_dir    = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / 'meta.npz'
    if meta_path.exists() and not force:
        print(f'[cache] Already exists at {cache_dir}  (use force=True to rebuild)')
        return

    print(f'[cache] Building voxel-grid cache (T={T}, H={H}, W={W}) ...')

    all_stems: list[str] = []
    all_labels: list[int] = []
    all_splits: list[str] = []

    for split in ('train', 'test'):
        stems = _read_trials(dataset_root, split)
        for stem in tqdm(stems, desc=f'{split}'):
            base = stem[:-len('.aedat')] if stem.endswith('.aedat') else stem
            ev_all = load_polarity_events(dataset_root / f'{base}.aedat')
            labels = _read_labels(dataset_root / f'{base}_labels.csv')
            for i, row in enumerate(labels):
                m = (ev_all['t'] >= row['start']) & (ev_all['t'] < row['end'])
                ev = {k: v[m] for k, v in ev_all.items()}
                voxel = events_to_voxel_grid(
                    ev['x'], ev['y'], ev['p'], ev['t'],
                    T=T, H=H, W=W,
                )
                key = f'{split}_{base}_{i}'
                np.savez_compressed(cache_dir / f'{key}.npz',
                                    voxel=voxel, label=np.int64(row['label']))
                all_stems.append(key)
                all_labels.append(row['label'])
                all_splits.append(split)

    np.savez(meta_path,
             stems=np.array(all_stems),
             labels=np.array(all_labels, dtype=np.int64),
             splits=np.array(all_splits))
    print(f'[cache] Done. {len(all_stems)} clips saved.')


class DVSGestureDataset(Dataset):

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = 'train',
        augment: bool = True,
        noise_std: float = 0.05,
        jitter_bins: int = 2,
        t_crop: int = 0,
    ):
        cache_dir = Path(cache_dir)
        meta = np.load(cache_dir / 'meta.npz', allow_pickle=True)
        mask = meta['splits'] == split
        self.stems  = meta['stems'][mask].tolist()
        self.labels = meta['labels'][mask].tolist()
        self.cache_dir   = cache_dir
        self.augment     = augment and split == 'train'
        self.noise_std   = noise_std
        self.jitter_bins = jitter_bins
        self.t_crop      = t_crop

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        data  = np.load(self.cache_dir / f'{self.stems[idx]}.npz')
        voxel = data['voxel'].astype(np.float32)
        label = int(data['label'])

        if self.t_crop > 0:
            T = voxel.shape[1]
            if T > self.t_crop:
                if self.augment:
                    start = np.random.randint(0, T - self.t_crop + 1)
                else:
                    start = (T - self.t_crop) // 2
                voxel = voxel[:, start:start + self.t_crop, :, :]

        if self.augment:
            voxel = self._augment(voxel)

        return torch.from_numpy(voxel), label

    def _augment(self, v: np.ndarray) -> np.ndarray:
        if self.jitter_bins > 0:
            shift = np.random.randint(-self.jitter_bins, self.jitter_bins + 1)
            if shift != 0:
                v = np.roll(v, shift, axis=1)

        if self.noise_std > 0:
            v = v + np.random.randn(*v.shape).astype(np.float32) * self.noise_std

        return v
