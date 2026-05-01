from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'train_v8c'))

from dvs_sota.dataset import DVSGestureDataset, build_cache
from dvs_sota.models import EfficientSpikingResNetV2

ROOT        = Path(__file__).parent
CACHE_DIR   = ROOT / 'cache_v8c_eval'
CKPT_PATH   = ROOT / 'checkpoints' / 'v8c_snn_best.pt'
DATASET_DIR = ROOT.parent / 'DvsGesture'

T_BINS_48 = 48
T_CROP    = 32
H = W     = 64
BATCH     = 32

CLASS_NAMES = [
    'Hand Clap', 'Right Hand Wave', 'Left Hand Wave',
    'Right Arm CW', 'Right Arm CCW', 'Left Arm CW', 'Left Arm CCW',
    'Arm Roll', 'Air Drums', 'Air Guitar', 'Other',
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')
    print(f'[device] {dev}')
    return dev


def count_params(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_evaluate(ckpt_path=None, dataset_dir=None, cache_dir=None,
                 rebuild_cache=False, batch_size=BATCH):
    device = get_device()

    ckpt_path   = Path(ckpt_path)   if ckpt_path   else CKPT_PATH
    dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR
    cache_dir   = Path(cache_dir)   if cache_dir   else CACHE_DIR

    if not ckpt_path.exists():
        print(f'[error] Checkpoint not found: {ckpt_path}')
        sys.exit(1)
    if not dataset_dir.exists():
        print(f'[error] Dataset path not found: {dataset_dir}')
        sys.exit(1)

    build_cache(dataset_dir, cache_dir, T=T_BINS_48, H=H, W=W, force=rebuild_cache)

    te  = DVSGestureDataset(cache_dir, 'test', augment=False, t_crop=T_CROP)
    tel = DataLoader(te, batch_size, shuffle=False, num_workers=0)
    print(f'[data] {len(te)} test clips  |  t_crop={T_CROP} (centre crop from T={T_BINS_48})')

    model = EfficientSpikingResNetV2(
        num_classes=11, T=T_CROP, beta=0.9,
        head_dropout=0.0, temporal_attn=True,
    ).to(device)
    n = count_params(model)
    print(f'[model] EfficientSpikingResNetV2  |  {n:,} parameters')

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['state'], strict=False)
    model.eval()
    print(f'[checkpoint] {ckpt_path.name}  '
          f'(saved at epoch {ck.get("epoch", "?")},  '
          f'recorded accuracy {ck.get("te_acc", 0)*100:.2f}%)')

    all_preds:  list[int] = []
    all_labels: list[int] = []

    print(f'\n[inference] Running on {device}...')
    t_start = time.time()
    with torch.no_grad():
        for voxel, labels in tqdm(tel, desc='  Evaluating', unit='batch'):
            voxel  = voxel.to(device)
            preds  = model(voxel).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    elapsed = time.time() - t_start

    n_total   = len(all_labels)
    n_correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy  = 100.0 * n_correct / n_total

    print(f'\n{"═"*54}')
    print(f'  Model     : EfficientSpikingResNetV2-T32-V8c')
    print(f'  Params    : {n:,}')
    print(f'  Test clips: {n_total}')
    print(f'  Correct   : {n_correct}')
    print(f'  Accuracy  : {accuracy:.2f}%')
    print(f'  Elapsed   : {elapsed:.1f}s')
    print(f'{"═"*54}\n')

    print(f'  {"Class":<22}  {"Recall":>7}  {"Correct":>7}  {"Total":>7}')
    print(f'  {"─"*22}  {"─"*7}  {"─"*7}  {"─"*7}')
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_total   = sum(l == cls_idx for l in all_labels)
        cls_correct = sum(p == l == cls_idx for p, l in zip(all_preds, all_labels))
        recall = 100.0 * cls_correct / cls_total if cls_total > 0 else 0.0
        flag   = ' ←' if recall < 80 else ''
        print(f'  {cls_name:<22}  {recall:>6.1f}%  {cls_correct:>7}  {cls_total:>7}{flag}')

    print()
    expected = 86.46
    delta = abs(accuracy - expected)
    if delta < 0.5:
        print(f'  ✓ Accuracy matches expected result ({expected}%).')
    else:
        print(f'  ⚠ Accuracy differs from expected {expected}% by {delta:.2f}pp.')
        print(f'    This may indicate a dataset or cache issue.')
    print()

    return accuracy


def main():
    ap = argparse.ArgumentParser(
        description='Evaluate EfficientSpikingResNetV2-T32-V8c on DVS128 Gesture.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--dataset',       default=None, metavar='PATH',
                    help='Root of the IBM DVS128 Gesture dataset.')
    ap.add_argument('--checkpoint',    default=None, metavar='PATH',
                    help='Path to v8c_snn_best.pt.')
    ap.add_argument('--cache_dir',     default=None, metavar='PATH',
                    help='Directory for the voxel cache.')
    ap.add_argument('--batch_size',    type=int, default=BATCH, metavar='N')
    ap.add_argument('--rebuild_cache', action='store_true',
                    help='Force rebuild of voxel cache.')
    args = ap.parse_args()
    run_evaluate(
        ckpt_path=args.checkpoint,
        dataset_dir=args.dataset,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
