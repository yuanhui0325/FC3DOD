"""Read-only statistics for the existing _C.bin / _F.bin formats."""
import argparse
import json
from pathlib import Path


def summarize(root, branches=1, baseline=None, legacy_shape_header=False):
    if not root.is_dir():
        raise FileNotFoundError(root)
    frames = {}
    ignored = 0
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if path.name.endswith('_Res_F.bin'):
            raise ValueError('Reconstruction streams are outside this detection release; use a detection-only directory')
        kind = None
        for suffix, category in [('_C.bin', 'geometry'), ('_F.bin', 'attribute')]:
            if path.name.endswith(suffix):
                kind = category
                frame = path.name[:-len(suffix)]
                break
        if kind is None:
            ignored += 1
            continue
        row = frames.setdefault(frame, {k: [] for k in ['geometry', 'attribute']})
        row[kind].append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size})
    if not frames:
        raise ValueError('No recognized real streams; NPY/NPZ/PLY files are not bitstreams.')
    for frame, row in frames.items():
        if len(row['geometry']) != branches or len(row['attribute']) != branches:
            raise ValueError('Incomplete/duplicate stream set for frame %s; expected %d geometry and attribute streams' % (frame, branches))
        for values in row.values():
            if any(x['bytes'] == 0 for x in values):
                raise ValueError('Empty stream for frame ' + frame)
        if legacy_shape_header and any(x['bytes'] <= 3 for x in row['attribute']):
            raise ValueError('Attribute stream lacks the 3-byte legacy spatial header')
    n = len(frames)
    totals = {kind: sum(x['bytes'] for row in frames.values() for x in row[kind]) for kind in ['geometry', 'attribute']}
    total = sum(totals.values())
    result = {'format': 'G-PCC geometry + deepCABAC attributes, NOT paper EE1',
              'sample_count': n, 'branches_per_frame': branches,
              'legacy_shape_header': legacy_shape_header,
              'total_bytes': totals, 'average_bytes_per_frame': {k: v / n for k, v in totals.items()},
              'average_total_bytes_per_frame': total / n,
              'average_total_KiB_per_frame': total / n / 1024,
              'average_total_decimal_kB_per_frame': total / n / 1000,
              'attribute_spatial_header_bytes_included': (3 if legacy_shape_header else 0) * branches * n,
              'average_attribute_CABAC_payload_bytes_per_frame': (totals['attribute'] - (3 if legacy_shape_header else 0) * branches * n) / n,
              'header_accounting': 'Actual file bytes include codec headers. Spatial shape is provided by model configuration; old three-byte prefixes are counted only with --legacy-shape-header. External configuration, frame/branch identifiers, calibration and transport framing are excluded.',
              'ignored_nonstream_files': ignored, 'frames': frames}
    if baseline is not None:
        if baseline <= 0:
            raise ValueError('Baseline bytes must be positive')
        result['user_supplied_baseline_bytes_per_frame'] = baseline
        result['baseline_to_stream_ratio'] = baseline / (total / n)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='one experiment root, no mixed runs')
    parser.add_argument('--branches', type=int, choices=[1, 3], default=1)
    parser.add_argument('--legacy-shape-header', action='store_true', help='Account for old three-byte-prefixed streams')
    parser.add_argument('--baseline-bytes-per-frame', type=float)
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.root, args.branches, args.baseline_bytes_per_frame, args.legacy_shape_header)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, str(error) + '\n')
    output = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + '\n')
    print(output)


if __name__ == '__main__':
    main()
