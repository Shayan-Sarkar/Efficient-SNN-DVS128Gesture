# DVS128 Gesture - SNN

Spiking neural network achieving **86.46% accuracy** on the IBM DVS128 Gesture dataset with only **320 K parameters**, trained via dual-teacher knowledge distillation.

Download Link: https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794

---

## Project Structure

```
final_report_bundle/
├── DvsGesture/          # Paste the raw IBM DVS128 Gesture dataset here (raw AEDAT extracted from DvsGesture.tar.gz + label CSVs)
├── evaluate_v8c/        # Evaluation pipeline — run inference on the trained checkpoint
└── train_v8c/           # Training pipeline — reproduce V8c from V8b warm-start
```

---

## Model Card

| Property | Value |
|---|---|
| Architecture | EfficientSpikingResNetV2 |
| Variant | T32-V8c |
| Parameters | 320,068 |
| Timesteps (T) | 32 (centre-cropped from 48-bin voxel) |
| Input resolution | 64 × 64 |
| Neuron model | Parametric LIF (learnable β, fast-sigmoid surrogate slope=25) |
| Channel attention | Squeeze-and-Excitation (reduction=4) |
| Temporal pooling | Soft attention (learned scalar scores, softmax) |
| Test accuracy | **86.46%** (249 / 288 clips) |
| Dataset | IBM DVS128 Gesture (11 classes) |

---

## Evaluation

### Requirements

```bash
pip install -r evaluate_v8c/requirements.txt
```

### Run

```bash
python evaluate_v8c/evaluate.py \
    --dataset   DvsGesture \
    --checkpoint evaluate_v8c/checkpoints/v8c_snn_best.pt
```

The script will:
1. Build a 48-bin voxel cache on first run (cached to `evaluate_v8c/cache_v8c_eval/`)
2. Centre-crop to T=32
3. Run inference and print per-class recall
4. Confirm whether accuracy matches the expected 86.46%

### Expected Output (abridged)

```
══════════════════════════════════════════════════════
  Model     : EfficientSpikingResNetV2-T32-V8c
  Params    : 320,068
  Test clips: 288
  Correct   : 249
  Accuracy  : 86.46%
══════════════════════════════════════════════════════

  Class                   Recall   Correct    Total
  ──────────────────────  ───────  ───────  ───────
  Hand Clap               91.7%       22       24
  Right Hand Wave        100.0%       24       24
  Left Hand Wave         100.0%       24       24
  Right Arm CW            70.8%       17       24  ←
  Right Arm CCW           79.2%       19       24  ←
  Left Arm CW             66.7%       16       24  ←
  Left Arm CCW            91.7%       22       24
  Arm Roll                95.8%       23       24
  Air Drums               75.0%       18       24  ←
  Air Guitar              66.7%       16       24  ←
  Other                  100.0%       48       48
```

### Optional Flags

| Flag | Default | Description |
|---|---|---|
| `--dataset PATH` | `../DvsGesture` | Root of raw DVS dataset |
| `--checkpoint PATH` | `evaluate_v8c/checkpoints/v8c_snn_best.pt` | Model checkpoint |
| `--cache_dir PATH` | `evaluate_v8c/cache_v8c_eval/` | Voxel cache directory |
| `--batch_size N` | 32 | Inference batch size |
| `--rebuild_cache` | off | Force rebuild of voxel cache |

---

## Training (V8c Reproduction)

V8c is the second cosine-annealing cycle starting from the V8b checkpoint (85.42%).

### Requirements

```bash
pip install -r train_v8c/requirements.txt
```

### Checkpoints Required

Place the following in `train_v8c/checkpoints/` before training:

| File | Role |
|---|---|
| `v8b_snn_best.pt` | Student warm-start (85.42%) |
| `compact_ann_best.pt` | Teacher 1 — EfficientResNet2DT (96.88%) |
| `ann_best.pt` | Teacher 2 — ResNet2DT (96.53%) |

### Run

```bash
cd train_v8c
python run_train.py
```

Training will:
1. Build the T=48 voxel cache (`voxel_cache_64x64_T48/`) if absent
2. Run 150 epochs of cosine-annealed AdamW with dual-teacher KD
3. Save the best checkpoint to `checkpoints/v8c_snn_best.pt`

### Key Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 150 |
| Base LR (late layers) | 5e-5 |
| LR scale — mid (blk2) | ×0.2 |
| LR scale — early (stem, blk1) | ×0.05 |
| Weight decay | 2e-3 |
| KD temperature | 4.0 |
| KD α (soft-label weight) | 0.7 |
| Feature alignment β | 0.2 |
| Mixup α | 0.2 |
| Temporal flip prob | 0.5 |
| Event dropout | 0.10 |
| LIF β (initial) | 0.9 (learnable) |

---

## Dataset

The `DvsGesture/` folder must contain the raw IBM DVS128 Gesture dataset:

```
DvsGesture/
├── user01_fluorescent/
│   ├── user01_fluorescent.aedat   # AEDAT 3.1 event stream
│   └── user01_fluorescent_labels.csv
├── user02_fluorescent_led/
│   ...
└── trials_to_train.txt / trials_to_test.txt
```

Train split: users 1–21 (per `trials_to_train.txt`)  
Test split: users 22–25 (per `trials_to_test.txt`)  
Classes: 11 (Hand Clap, Right/Left Hand Wave, Right/Left Arm CW/CCW, Arm Roll, Air Drums, Air Guitar, Other)

---

## Citation

If you use this work, please cite the IBM DVS128 Gesture dataset:

> Amir et al., "A Low Power, Fully Event-Based Gesture Recognition System," CVPR 2017.
