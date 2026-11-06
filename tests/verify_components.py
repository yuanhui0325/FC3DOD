"""Small real backend checks; CPU codecs/helpers only, never CPU detection."""
import argparse
import json
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch
    from pcgc.data_utils import read_ply_ascii_geo, write_ply_ascii_geo, sort_spare_tensor
    from pcgc.gpcc import gpcc_encode, gpcc_decode
    from tools.stream_stats import summarize
    result = {'python': sys.version.split()[0], 'torch': torch.__version__,
              'cuda_available': torch.cuda.is_available(), 'checks': {}}
    with tempfile.TemporaryDirectory(prefix='fc3dod checks ') as directory:
        root = Path(directory)
        coords = np.array([[0, 1, 2], [3, 4, 5], [7, 9, 11], [3, 4, 7]], dtype=np.int32)
        ply = root / 'input cloud.ply'
        write_ply_ascii_geo(ply, coords)
        assert np.array_equal(read_ply_ascii_geo(ply), coords)
        empty = root / 'empty.ply'
        write_ply_ascii_geo(empty, np.empty((0, 3)))
        assert read_ply_ascii_geo(empty).shape == (0, 3)
        result['checks']['PLY_nonempty_empty_space_path'] = 'passed'
        # Verify the actual sorting helper preserves coordinate-to-attribute correspondence.
        class TensorRows:
            def __init__(self):
                self.indices = torch.tensor([[0, 2, 1, 3], [0, 1, 0, 1], [0, 0, 2, 0]])
                self.features = torch.tensor([[21.], [10.], [2.]])
            def replace_feature(self, values):
                self.features = values
                return self
        rows = TensorRows()
        pairs = {tuple(c.tolist()): f.item() for c, f in zip(rows.indices, rows.features)}
        rows = sort_spare_tensor(rows)
        assert all(pairs[tuple(c.tolist())] == f.item() for c, f in zip(rows.indices, rows.features))
        result['checks']['original_sort_coordinate_attribute_alignment'] = 'passed'
        try:
            import deepCABAC
            original = np.array([[-2., 0., 1.], [3., 5., -7.]], dtype=np.float32)
            encoder = deepCABAC.Encoder()
            encoder.encodeWeightsRD(original, 1., 1., 0.)
            stream = encoder.finish()
            decoder = deepCABAC.Decoder()
            decoder.getStream(stream)
            decoded = decoder.decodeWeights()
            assert np.array_equal(original, decoded)
            result['checks']['deepCABAC_real_roundtrip'] = {'status': 'passed', 'stream_bytes': len(stream), 'scalar_count': original.size, 'max_error': float(np.max(np.abs(original-decoded))), 'scope': 'Synthetic integer attributes, not paper end-to-end coding'}
            floating = np.array([[0.11, 1.23, -2.21], [3.77, -0.51, 0.0]], dtype=np.float32)
            q = 4.0
            quantized = np.round(floating * q)
            encoder = deepCABAC.Encoder()
            encoder.encodeWeightsRD(quantized, 1., 1., 0.)
            stream = encoder.finish()
            decoder = deepCABAC.Decoder(); decoder.getStream(stream)
            decoded = decoder.decodeWeights() / q
            assert np.array_equal(decoded * q, quantized)
            error = np.abs(floating - decoded)
            assert error.max() <= 0.5 / q + 1e-6
            result['checks']['deepCABAC_attribute_quantization_probe'] = {'status': 'passed',
                'scalar_count': floating.size, 'stream_bytes': len(stream),
                'quantization_parameter': q, 'interval': 1., 'step_size': 1., 'lambda': 0.,
                'max_error': float(error.max()), 'mean_error': float(error.mean()),
                'scope': 'Round-and-scale attribute backend probe only, not full sparse codec or paper mode'}

        except (ImportError, RuntimeError) as error:
            result['checks']['deepCABAC_real_roundtrip'] = {'status': 'blocked', 'reason': type(error).__name__}
        try:
            binary = root / 'geometry stream.bin'
            decoded_ply = root / 'decoded cloud.ply'
            gpcc_encode(ply, binary)
            gpcc_decode(binary, decoded_ply)
            decoded = read_ply_ascii_geo(decoded_ply)
            assert set(map(tuple, coords)) == set(map(tuple, decoded))
            result['checks']['G_PCC_real_roundtrip'] = {'status': 'passed', 'stream_bytes': binary.stat().st_size, 'point_count': len(coords), 'coordinate_set_equal': True, 'scope': 'Geometry only; attributes require the original common sort order'}
        except (FileNotFoundError, RuntimeError, __import__('subprocess').CalledProcessError) as error:
            result['checks']['G_PCC_real_roundtrip'] = {'status': 'blocked', 'reason': type(error).__name__}
        # Synthetic file lengths test accounting, not compression performance.
        stats_root = root / 'stats'; stats_root.mkdir()
        (stats_root / '000000_C.bin').write_bytes(b'g' * 11)
        (stats_root / '000000_F.bin').write_bytes(b'a' * 13)
        (stats_root / 'debug.npy').write_bytes(b'x' * 99)
        (stats_root / 'keep.ply').write_text('preserve')
        stats = summarize(stats_root)
        assert stats['sample_count'] == 1 and stats['average_total_bytes_per_frame'] == 24
        assert stats['attribute_spatial_header_bytes_included'] == 0
        assert summarize(stats_root, legacy_shape_header=True)['attribute_spatial_header_bytes_included'] == 3
        assert (stats_root / 'keep.ply').read_text() == 'preserve'
        (stats_root / '000001_C.bin').write_bytes(b'partial')
        try:
            summarize(stats_root)
        except ValueError:
            pass
        else:
            raise AssertionError('Partial streams must be rejected')
        (stats_root / '000000_Res_F.bin').write_bytes(b'not public')
        try:
            summarize(stats_root)
        except ValueError as error:
            assert 'Reconstruction streams' in str(error)
        else:
            raise AssertionError('Mixed reconstruction outputs must be rejected')
        result['checks']['statistics_header_exclusion_incomplete_non_destructive'] = 'passed (synthetic lengths only)'
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
