# FC3DOD

Code for **Feature Compression for Cloud-Edge Multimodal 3D Object Detection**, IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(7), 8411–8428, 2026.

Chongzhen Tian, Zhengxin Li, Hui Yuan, Raouf Hamzaoui, Liquan Shen, and Sam Kwong.

[Paper / DOI](https://doi.org/10.1109/TPAMI.2026.3669471) · [Repository](https://github.com/yuanhui0325/FC3DOD)

FC3DOD compresses sparse multimodal features between an edge feature extractor and a cloud detector. **T-FFC** transmits the basic compressed feature. **A-FFC** additionally transmits features from F3 and F4 using guidance from the basic branch.

## Features

- PENet virtual-point preparation and KITTI multimodal preprocessing.
- VirConv-based T-FFC/A-FFC with masked Transformer enhancement, feature selection, backward mapping and cloud detection.
- G-PCC geometry coding and CABAC attribute coding during evaluation.
- T-FFC/A-FFC model checkpoints and KITTI train/validation splits.

Use one frame per batch with a single view (`--batch_size 1`). The basic feature branch has four channels, and each A-FFC additional branch has two channels.

## Installation

Detection requires a working NVIDIA GPU, a compatible Python/PyTorch/CUDA/spconv toolchain and the custom CUDA extensions. CPU detection is not supported.

Reference environment used for component checks:

| Component | Version / result |
|---|---|
| Python | 3.9.13 |
| PyTorch | 1.9.1+cu111 |
| CUDA runtime reported by PyTorch | 11.1 |
| spconv | spconv-cu111 2.1.25 |
| NumPy | 1.23.0 |
| deepCABAC | 0.1.0 |

Install a compatible PyTorch/spconv pair for your CUDA toolkit and driver first, then run:

```bash
cd /path/to/FC3DOD
python -m pip install -r requirements.txt -r requirements-build.txt
(cd cabac && python -m pip install --no-build-isolation --no-deps .)
python -m pip install --no-build-isolation --no-deps -e .
```

`--no-deps` preserves the chosen CUDA-specific spconv wheel. The root installer builds the detection CUDA operators; it requires a compatible CUDA toolkit/compiler. The optional deformable-convolution installer is `pcdet/ops/dcn/setup.py`.

For detector geometry coding, provide a compatible G-PCC `tmc3` executable:

```bash
export FC3DOD_TMC3=/absolute/path/to/tmc3
```

The G-PCC wrapper uses v12-style command-line options. Set `FC3DOD_TMC3` to a compatible executable.

## KITTI preparation

The supplied split contains **3712 training / 3769 validation frames**, with no overlap. Detection is evaluated using KITTI AP_R40 (40 recall sampling positions).

```text
data/kitti/
  ImageSets/{train,val,test}.txt
  training/
    velodyne/
    image_2/
    calib/
    label_2/
    planes/
    velodyne_depth/           # generated multimodal NPY points
  kitti_infos_train.pkl
  kitti_infos_val.pkl
  kitti_dbinfos_train_mm.pkl
  gt_database_mm/
```

Place KITTI data under this directory or use an external root containing the supplied `ImageSets` files. Dataset paths resolve relative to the repository. Override with `--set DATA_CONFIG.DATA_PATH /absolute/path/to/kitti` for train/test.

PENet generates `velodyne_depth/<frame>.npy` from images, LiDAR and calibration. It requires a separate compatible checkpoint. From the PENet directory:

```bash
cd tools/PENet
CUDA_VISIBLE_DEVICES=0 python main.py \
  --evaluate /absolute/path/to/penet_checkpoint.pth.tar \
  --detpath /absolute/path/to/kitti/training --workers 0
```

Use the corresponding detection split when preparing testing data. The current loader assumes cropped depth 352×1216 and generates eight point attributes. [Input-channel documentation](docs/input_features.md) describes their meaning.

After generating virtual points, create multimodal metadata and the ground-truth database from the repository root:

```bash
python tools/prepare_kitti.py --data-path /absolute/path/to/kitti --workers 4
```

Metadata generation processes the dataset and creates the files required for training and evaluation.



## Evaluation

Run from the repository root with the corresponding checkpoint. These two-frame examples save smoke predictions and do not calculate KITTI AP:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/models/kitti/FC3DOD-T-FFC.yaml \
  --ckpt weights/tffc.pth --batch_size 1 --workers 0 --max_frames 2 \
  --set DATA_CONFIG.DATA_PATH /absolute/path/to/kitti

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/models/kitti/FC3DOD-A-FFC.yaml \
  --ckpt weights/affc.pth --batch_size 1 --workers 0 --max_frames 2 \
  --set DATA_CONFIG.DATA_PATH /absolute/path/to/kitti
```

Remove `--max_frames 2` for full validation and AP_R40 evaluation. The batch size must remain 1. Evaluation uses `model.eval()` and `torch.no_grad()` with real encoding/decoding.

Default predictions and logs are under `output/models/kitti/<config>/<extra_tag>/eval/`. Use `--output_dir /absolute/path` to change the output root.

## Training

During training, floating-point branch features pass directly to the cloud detection modules. G-PCC/CABAC encoding and decoding are applied during evaluation.

The following commands demonstrate a one-epoch training run. Set `--epochs` for your training schedule:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/models/kitti/FC3DOD-T-FFC.yaml \
  --batch_size 1 --workers 0 --epochs 1 \
  --set DATA_CONFIG.DATA_PATH /absolute/path/to/kitti

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/models/kitti/FC3DOD-A-FFC.yaml \
  --batch_size 1 --workers 0 --epochs 1 \
  --set DATA_CONFIG.DATA_PATH /absolute/path/to/kitti
```

Use `--pretrained_model /path/to/compatible_checkpoint.pth` for initialization or `--ckpt /path/to/training_checkpoint.pth` for resuming. Checkpoints are saved under the experiment's `ckpt/` directory.

## Encoding, decoding and rate statistics

The detector writes real streams during evaluation. Set the output root before evaluation:

```bash
export FC3DOD_STREAM_ROOT=/absolute/path/to/streams
```

Defaults are under `output/streams/tffc/r=…/k=…` and `output/streams/affc/r=…/k=…/{bev,roi3,roi4}`. Use separate experiment roots to avoid mixing runs or overwriting frames.

For exported sparse features, the CUDA CLI invokes the detector's existing codec. Coordinates must be integer `(N,4)` `[batch,z,y,x]`, aligned with `(N,C)` attributes; they must be unique and in bounds. Both endpoints supply the same fixed grid; spatial dimensions are not transmitted in the application attribute header.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/feature_codec.py roundtrip --variant tffc \
  --coords /path/to/coords.npy --features /path/to/features.npy \
  --shape 10 200 176 --stream-dir output/codec_probe --frame 000000 \
  --quantization-parameter 1 --interval 1 --step-size 1 --device cuda:0

CUDA_VISIBLE_DEVICES=0 python tools/feature_codec.py decode --variant tffc \
  --shape 10 200 176 --stream-dir output/codec_probe --frame 000000 \
  --quantization-parameter 1 --device cuda:0

python tools/stream_stats.py output/codec_probe --branches 1
python tools/stream_stats.py /path/to/one/affc/experiment --branches 3
```

`--variant affc` wraps one branch codec; encode the three branches separately using their actual settings. The retained A-FFC uses basic step 1 and additional-branch step 0.1; do not assume one CLI default reproduces every branch setting. The current grid configuration gives F4 `[10,200,176]` and F3 `[21,400,352]`; other voxel configurations require their own grids.

Geometry `_C.bin` is G-PCC; attributes `_F.bin` are CABAC. Statistics report complete-frame counts and mean geometry/attribute/total **bytes**, decimal kB and KiB. Decoded NPY/PLY inspection files are excluded. To decode or account for old three-byte-prefixed attribute files, explicitly use `--legacy-shape-header`; do not mix old and new formats.

[Rate accounting](docs/rate_accounting.md) describes the distinction between feature reduction, tensor memory and encoded bitstream size, along with stream headers and reporting units.

## Component tests

```bash
python tests/verify_components.py --json-out output/component_validation.json
python tests/verify_codec_contracts.py
python tests/verify_training_codec_gate.py
```

The component checks require deepCABAC, PyTorch and, for the G-PCC probe, `FC3DOD_TMC3`.

## Citation and licenses

```bibtex
@article{tian2026fc3dod,
  title={Feature Compression for Cloud-Edge Multimodal 3D Object Detection},
  author={Tian, Chongzhen and Li, Zhengxin and Yuan, Hui and Hamzaoui, Raouf and Shen, Liquan and Kwong, Sam},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}, volume={48}, number={7}, pages={8411--8428},
  doi={10.1109/TPAMI.2026.3669471}
}
```

Acknowledgments: VirConv, OpenPCDet, PENet, KITTI evaluation code, deepCABAC and MPEG G-PCC/TMC13.

The inherited [top-level license](LICENSE) and each component's original license/notices are retained. They must be read separately; the top-level license does not override third-party terms. In particular, [deepCABAC's notice](cabac/LICENSE) specifies testing functionality and does not grant patent rights. See [third-party notices](docs/third_party.md).
