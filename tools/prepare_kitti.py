"""Generate KITTI multimodal metadata through the existing KittiDatasetMM utility."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--cfg-file', default='cfgs/models/kitti/FC3DOD-T-FFC.yaml')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.kitti.kitti_dataset_mm import create_kitti_infos
    cfg_from_yaml_file(args.cfg_file, cfg)
    data = args.data_path.expanduser().resolve()
    cfg.DATA_CONFIG.DATA_PATH = str(data)
    # Existing utility intentionally starts with empty metadata while producing infos.
    cfg.DATA_CONFIG.INFO_PATH = {'train': [], 'test': []}
    create_kitti_infos(cfg.DATA_CONFIG, ['Car', 'Pedestrian', 'Cyclist'], data, data, args.workers)


if __name__ == '__main__':
    main()
