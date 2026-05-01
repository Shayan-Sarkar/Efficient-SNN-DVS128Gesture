from __future__ import annotations
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT        = Path(__file__).parent
DATASET_DIR = ROOT.parent / 'DvsGesture'
CACHE_DIR_T48 = ROOT / 'voxel_cache_64x64_T48'
CKPT_DIR    = ROOT / 'checkpoints'
RESULTS_CSV = ROOT / 'experiments_sota.csv'
T_BINS_48   = 48
T_BINS_32   = 32
H = W = 64
BATCH = 32

V8B_SNN_CFG = dict(
    finetune_epochs = 150,
    base_lr         = 8e-5,
    lr_scale_mid    = 0.2,
    lr_scale_early  = 0.05,
    weight_decay    = 2e-3,
    beta            = 0.9,
    kd_temp         = 4.0,
    kd_alpha        = 0.7,
    feat_beta       = 0.2,
    event_drop_p    = 0.10,
    mixup_alpha     = 0.2,
    head_dropout    = 0.3,
    tflip_p         = 0.5,
    t_crop          = 32,
)

V8C_SNN_CFG = dict(
    finetune_epochs = 150,
    base_lr         = 5e-5,
    lr_scale_mid    = 0.2,
    lr_scale_early  = 0.05,
    weight_decay    = 2e-3,
    beta            = 0.9,
    kd_temp         = 4.0,
    kd_alpha        = 0.7,
    feat_beta       = 0.2,
    event_drop_p    = 0.10,
    mixup_alpha     = 0.2,
    head_dropout    = 0.3,
    tflip_p         = 0.5,
    t_crop          = 32,
)


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


def save_result(row: dict):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not RESULTS_CSV.exists()
    with RESULTS_CSV.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)
    print(f'[result] -> {RESULTS_CSV}')


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct = total = 0
    for v, l in loader:
        v, l = v.to(device), l.to(device)
        correct += (model(v).argmax(1) == l).sum().item()
        total += len(l)
    return correct / total


def run_preprocess_t48():
    from dvs_sota.dataset import build_cache
    build_cache(DATASET_DIR, CACHE_DIR_T48, T=T_BINS_48, H=H, W=W)


def _make_loaders_t48_crop(t_crop: int = 32, augment_train: bool = True):
    from dvs_sota.dataset import DVSGestureDataset
    tr = DVSGestureDataset(CACHE_DIR_T48, 'train', augment=augment_train,
                           t_crop=t_crop)
    te = DVSGestureDataset(CACHE_DIR_T48, 'test',  augment=False,
                           t_crop=t_crop)
    trl = DataLoader(tr, BATCH, shuffle=True,  num_workers=0)
    tel = DataLoader(te, BATCH, shuffle=False, num_workers=0)
    print(f'[data-T48-crop] train={len(tr)}  test={len(te)}  '
          f't_crop={t_crop}  batch={BATCH}')
    return trl, tel


def _make_layer_param_groups(model, base_lr, lr_scale_mid, lr_scale_early, weight_decay):
    early_names = {'stem_conv', 'stem_in', 'stem_lif', 'blk1'}
    mid_names   = {'blk2'}

    early_params, mid_params, late_params = [], [], []
    for name, param in model.named_parameters():
        prefix = name.split('.')[0]
        if prefix in early_names:
            early_params.append(param)
        elif prefix in mid_names:
            mid_params.append(param)
        else:
            late_params.append(param)

    return [
        {'params': early_params, 'lr': base_lr * lr_scale_early,
         'weight_decay': weight_decay, 'name': 'early'},
        {'params': mid_params,   'lr': base_lr * lr_scale_mid,
         'weight_decay': weight_decay, 'name': 'mid'},
        {'params': late_params,  'lr': base_lr,
         'weight_decay': weight_decay, 'name': 'late'},
    ]


def _run_v6_v7_training(
    cfg: dict,
    T_bins: int,
    trl, tel,
    warmstart_ckpt_path,
    save_ckpt_name: str,
    phase_label: str,
    model_label: str,
    device,
):
    from dvs_sota.models import EfficientResNet2DT, ResNet2DT, EfficientSpikingResNetV2

    base_lr        = cfg['base_lr']
    lr_scale_mid   = cfg['lr_scale_mid']
    lr_scale_early = cfg['lr_scale_early']
    wd             = cfg['weight_decay']
    kd_temp        = cfg['kd_temp']
    alpha          = cfg['kd_alpha']
    f_beta         = cfg['feat_beta']
    drop_p         = cfg['event_drop_p']
    mx_alpha       = cfg['mixup_alpha']
    tflip_p        = cfg['tflip_p']
    epochs         = cfg['finetune_epochs']

    t1_ckpt  = CKPT_DIR / 'compact_ann_best.pt'
    teacher1 = EfficientResNet2DT(num_classes=11).to(device)
    ck1 = torch.load(t1_ckpt, map_location=device)
    teacher1.load_state_dict(ck1['state'])
    teacher1.eval()
    print(f'[teacher1] EfficientResNet2DT  '
          f'epoch={ck1["epoch"]}  te_acc={ck1["te_acc"]*100:.2f}%')

    t2_ckpt  = CKPT_DIR / 'ann_best.pt'
    teacher2 = ResNet2DT(num_classes=11).to(device)
    ck2 = torch.load(t2_ckpt, map_location=device)
    teacher2.load_state_dict(ck2['state'])
    teacher2.eval()
    print(f'[teacher2] ResNet2DT  '
          f'epoch={ck2["epoch"]}  te_acc={ck2["te_acc"]*100:.2f}%')

    _t1_gap_feats: list = []

    def _gap_hook(mod, inp, out):
        _t1_gap_feats.clear()
        _t1_gap_feats.append(out.detach().flatten(1))

    hook_handle = teacher1.gap.register_forward_hook(_gap_hook)

    student = EfficientSpikingResNetV2(
        num_classes=11, T=T_bins,
        beta=cfg['beta'],
        head_dropout=cfg['head_dropout'],
        temporal_attn=True,
    ).to(device)

    ws_ck = torch.load(warmstart_ckpt_path, map_location=device)
    missing, unexpected = student.load_state_dict(ws_ck['state'], strict=False)
    print(f'[warm-start] {warmstart_ckpt_path.name}  '
          f'(epoch={ws_ck["epoch"]}  te_acc={ws_ck["te_acc"]*100:.2f}%)')
    if missing:
        print(f'  missing keys (re-init): {missing}')

    n = count_params(student)
    print(f'[student] {model_label}  T={T_bins}  params={n:,}')
    print(f'[per-layer LR]  '
          f'early={base_lr*lr_scale_early:.1e}  '
          f'mid={base_lr*lr_scale_mid:.1e}  '
          f'late={base_lr:.1e}')
    print(f'[temporal-flip] p={tflip_p}')

    param_groups = _make_layer_param_groups(
        student, base_lr, lr_scale_mid, lr_scale_early, wd
    )
    opt   = AdamW(param_groups)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit  = nn.CrossEntropyLoss()

    best = 0.0
    t0   = time.time()

    for ep in range(1, epochs + 1):
        student.train(); teacher1.eval(); teacher2.eval()
        tr_correct = tr_total = 0
        tr_loss_sum = 0.0

        for v, l in trl:
            v, l = v.to(device), l.to(device)
            B = v.size(0)
            opt.zero_grad()

            if tflip_p > 0 and torch.rand(1).item() < tflip_p:
                v = v.flip(2)

            with torch.no_grad():
                t1_logits = teacher1(v)
                t1_feat   = _t1_gap_feats[0].reshape(B, T_bins, -1).mean(1)
                t2_logits = teacher2(v)

            lam = float(np.random.beta(mx_alpha, mx_alpha))
            idx = torch.randperm(B, device=device)
            v_mix = lam * v + (1.0 - lam) * v[idx]

            t1_soft_i = F.softmax(t1_logits      / kd_temp, dim=1)
            t1_soft_j = F.softmax(t1_logits[idx] / kd_temp, dim=1)
            t2_soft_i = F.softmax(t2_logits      / kd_temp, dim=1)
            t2_soft_j = F.softmax(t2_logits[idx] / kd_temp, dim=1)
            t_soft_mix = (lam * 0.5 * (t1_soft_i + t2_soft_i) +
                          (1.0 - lam) * 0.5 * (t1_soft_j + t2_soft_j))
            t1_feat_mix = lam * t1_feat + (1.0 - lam) * t1_feat[idx]

            if drop_p > 0.0:
                mask  = torch.rand_like(v_mix) > drop_p
                v_aug = v_mix * mask.float()
            else:
                v_aug = v_mix

            s_logits, s_feat = student(v_aug, return_feat=True)

            s_log_soft = F.log_softmax(s_logits / kd_temp, dim=1)
            kl_loss    = F.kl_div(s_log_soft, t_soft_mix,
                                  reduction='batchmean') * (kd_temp ** 2)
            s_fn      = F.normalize(s_feat,      dim=1)
            t_fn      = F.normalize(t1_feat_mix, dim=1)
            feat_loss = F.mse_loss(s_fn, t_fn)
            ce_loss   = (lam * crit(s_logits, l) +
                         (1.0 - lam) * crit(s_logits, l[idx]))
            loss = alpha * kl_loss + f_beta * feat_loss + (1 - alpha - f_beta) * ce_loss

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            tr_loss_sum += loss.item() * B
            tr_correct  += (s_logits.argmax(1) == l).sum().item()
            tr_total    += B

        sched.step()
        te_acc = eval_epoch(student, tel, device)
        lr_now = opt.param_groups[-1]['lr']
        tr_acc = tr_correct / tr_total
        print(f'Ep {ep:3d}/{epochs}  '
              f'loss={tr_loss_sum/tr_total:.4f}  tr={tr_acc*100:.1f}%  '
              f'te={te_acc*100:.2f}%  lr={lr_now:.1e}  '
              f'({(time.time()-t0)/60:.1f}m)  best={best*100:.2f}%', flush=True)
        if te_acc > best:
            best = te_acc
            torch.save({'epoch': ep, 'state': student.state_dict(),
                        'te_acc': te_acc},
                       CKPT_DIR / save_ckpt_name)
            print(f'  ^ new best {te_acc*100:.2f}%')

    hook_handle.remove()
    print(f'\n[{phase_label}] Best test accuracy: {best*100:.2f}%')
    save_result({
        'phase': phase_label,
        'representation': f'voxel_grid_T{T_bins}_{H}x{W}',
        'model': model_label,
        'params': n,
        'best_val_acc': '',
        'best_test_acc': round(best * 100, 2),
        'epochs': epochs,
        'notes': (
            f'{model_label}: warmstart {warmstart_ckpt_path.name}; T={T_bins}; '
            f'per-layer LR (early={base_lr*lr_scale_early:.0e} '
            f'mid={base_lr*lr_scale_mid:.0e} late={base_lr:.0e}); '
            f'temporal-flip p={tflip_p}; dual-teacher KD; GAP feat-KD; '
            f'alpha={alpha} f_beta={f_beta}; wd={wd}; BPTT T={T_bins}'
        ),
    })
    return best


def run_train_v8c_snn():
    device = get_device()

    if not CACHE_DIR_T48.exists():
        print('[V8c] T=48 cache not found — building ...')
        run_preprocess_t48()

    t_crop = V8C_SNN_CFG['t_crop']
    trl, tel = _make_loaders_t48_crop(t_crop=t_crop, augment_train=True)

    warmstart_path = CKPT_DIR / 'v8b_snn_best.pt'
    if not warmstart_path.exists():
        raise FileNotFoundError(
            'v8b_snn_best.pt not found. Provide it in the checkpoints/ directory.')
    print(f'[V8c] warmstart from {warmstart_path.name}')
    print(f'[V8c] second cosine cycle  base_lr={V8C_SNN_CFG["base_lr"]:.0e}  '
          f't_crop={t_crop}')

    _run_v6_v7_training(
        cfg=V8C_SNN_CFG,
        T_bins=t_crop,
        trl=trl, tel=tel,
        warmstart_ckpt_path=warmstart_path,
        save_ckpt_name='v8c_snn_best.pt',
        phase_label='efficiency_frontier_v8c',
        model_label=f'EfficientSpikingResNetV2-T{t_crop}-V8c-crop',
        device=device,
    )


if __name__ == '__main__':
    run_train_v8c_snn()
