# evaluate_v8c

Evaluate the V8c SNN checkpoint (`EfficientSpikingResNetV2`, ~320K params) on the DVS128 Gesture test set.

## Requirements

```
pip install -r requirements.txt
```

## Usage

```bash
python evaluate.py
```

By default this uses:
- Checkpoint: `checkpoints/v8c_snn_best.pt`
- Dataset: `../DvsGesture/`
- Cache: `cache_v8c_eval/` (built automatically on first run)

### Options

```bash
python evaluate.py --ckpt path/to/checkpoint.pt \
                   --dataset path/to/DvsGesture \
                   --cache path/to/cache_dir \
                   --rebuild-cache
```

## Expected result

**86.46% test accuracy** (V8c, second cosine cycle from V8b 85.42%)

Architecture: EfficientSpikingResNetV2 with SE channel attention, temporal soft attention, ~320K parameters. Evaluated with T=32 centre-crop from T=48 voxel grids at 64x64 spatial resolution.
