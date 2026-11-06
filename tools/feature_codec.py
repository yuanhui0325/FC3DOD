"""Invoke an existing EntropyEncDec; legacy G-PCC/deepCABAC, not paper EE1."""
import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['encode', 'decode', 'roundtrip'])
    parser.add_argument('--variant', choices=['tffc', 'affc'], default='tffc')
    parser.add_argument('--coords', type=Path, help='NPY integer (N,4), [batch,z,y,x]')
    parser.add_argument('--features', type=Path, help='NPY float (N,C), same point order')
    parser.add_argument('--shape', type=int, nargs=3, required=True, metavar=('Z', 'Y', 'X'), help='Fixed model grid known to both endpoints; not transmitted')
    parser.add_argument('--legacy-shape-header', action='store_true', help='Read old three-byte-header streams; decode only')
    parser.add_argument('--stream-dir', type=Path, required=True)
    parser.add_argument('--frame', default='000000')
    parser.add_argument('--quantization-parameter', type=float, default=1)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--step-size', type=float, default=1)
    parser.add_argument('--lambda-rd', type=float, default=0)
    parser.add_argument('--device', default='cuda:0', help='CUDA only; respects CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    if args.legacy_shape_header and args.action != 'decode':
        parser.error('--legacy-shape-header is supported only for decoding old streams')
    if Path(args.frame).name != args.frame or args.frame in {'.', '..'}:
        parser.error('--frame must be a single filename component')
    if args.quantization_parameter <= 0 or args.interval <= 0 or args.step_size <= 0:
        parser.error('quantization-parameter, interval and step-size must be positive')
    import numpy as np
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.exit(2, 'A working CUDA GPU and compatible spconv/custom operators are required.\n')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    module_name = 'spconv_backbone_' + ('single_branch_without_model_split' if args.variant == 'tffc' else 'three_branches_without_model_split')
    module = importlib.import_module('pcdet.models.backbones_3d.' + module_name)
    args.stream_dir.mkdir(parents=True, exist_ok=True)
    codec = module.EntropyEncDec(args.quantization_parameter, args.interval, args.step_size,
                                args.lambda_rd, str(args.stream_dir.resolve()) + '/',
                                spatial_shape=args.shape, legacy_shape_header=args.legacy_shape_header)
    codec.decode_device = device
    expected = None
    if args.action != 'decode':
        if args.coords is None or args.features is None or args.shape is None:
            parser.error('encode/roundtrip require --coords, --features and --shape')
        coords = np.load(args.coords, allow_pickle=False)
        features = np.load(args.features, allow_pickle=False)
        if coords.ndim != 2 or coords.shape[1] != 4 or features.ndim != 2 or len(coords) != len(features):
            parser.error('Expected aligned coordinates (N,4) and attributes (N,C)')
        if not np.isfinite(coords).all() or not np.equal(coords, np.round(coords)).all():
            parser.error('Coordinates must be finite integers')
        if (coords[:, 0] != 0).any() or (coords[:, 1:] < 0).any() or (coords[:, 1:] >= np.array(args.shape)).any():
            parser.error('Coordinates must be in bounds with batch index zero')
        if len(np.unique(coords, axis=0)) != len(coords):
            parser.error('Duplicate coordinates cannot be represented by the legacy geometry codec')
        sparse = module.spconv.SparseConvTensor(torch.tensor(features, dtype=torch.float32, device=device),
                                               torch.tensor(coords, dtype=torch.int32, device=device), args.shape, 1)
        codec.encode(sparse, args.frame)
        # encode sorts coordinates and features together using the original helper.
        expected = sparse
    if args.action != 'encode':
        decoded = codec.decode(args.frame)
        coords_out = decoded.indices.cpu().numpy()
        features_out = decoded.features.cpu().numpy()
        np.save(args.stream_dir / (args.frame + '_decoded_coords.npy'), coords_out)
        np.save(args.stream_dir / (args.frame + '_decoded_features.npy'), features_out)
        if expected is not None:
            if not np.array_equal(expected.indices.cpu().numpy(), coords_out):
                raise RuntimeError('Geometry/order mismatch')
            reference = expected.features.cpu().numpy()
            error = np.abs(reference - features_out)
            print(json.dumps({'point_count': len(coords_out), 'coordinate_order_equal': True,
                              'max_attribute_error': float(error.max()),
                              'mean_attribute_error': float(error.mean()),
                              'quantization_parameter': args.quantization_parameter,
                              'interval': args.interval, 'step_size': args.step_size}))


if __name__ == '__main__':
    main()
