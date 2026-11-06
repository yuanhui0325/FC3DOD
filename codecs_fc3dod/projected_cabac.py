"""Projected lattice + patch-differential CABAC reference codec.

Version 2 predicts attributes from the same lattice cell in the previous patch.
Fixed grid shape is supplied by decoder configuration, not sent in the stream.
Version 1 decoding retains the older DC predictor and stored shape.
"""
import json
import struct
from pathlib import Path
import numpy as np
from pcgc.codec_config import validate_shape

MAGIC = b'FCPC2\0'
LEGACY_MAGIC = b'FCPC1\0'


def _angle(shape, theta):
    shape = validate_shape(shape)
    limit = 90. if shape[0] == 1 else np.rad2deg(np.arctan(1/(shape[0]-1)))
    theta = limit/2 if theta is None else theta
    if not np.isfinite(theta) or not 0 < theta < limit:
        raise ValueError('Require 0 < theta < atan(1/(Z-1)) in degrees.')
    angle = np.deg2rad(theta)
    tolerance = 32*np.finfo(np.float64).eps*max(shape[2], 1)
    if (shape[0] > 1 and np.sin(angle) <= tolerance) or (shape[2] > 1 and np.cos(angle)-(shape[0]-1)*np.sin(angle) <= tolerance):
        raise ValueError('Rotation angle is too close to a collision bound at float64 precision.')
    return float(theta)


def lattice_projection(coords, shape, theta):
    shape = validate_shape(shape)
    theta = _angle(shape, theta)
    z, y, x = np.asarray(coords)[:, 1:].T
    angle = np.deg2rad(theta)
    xy = np.column_stack((x*np.cos(angle)+z*np.sin(angle), y))
    slots = np.column_stack((y, x*shape[0]+z)).astype(np.int64)
    return slots, xy


def lattice_inverse(slots, shape):
    shape = validate_shape(shape)
    slots = np.asarray(slots)
    if slots.ndim != 2 or slots.shape[1] != 2 or not np.isfinite(slots).all() or (slots != np.floor(slots)).any():
        raise ValueError('Expected integer projected slots (N,2).')
    if (slots < 0).any() or (slots[:, 0] >= shape[1]).any() or (slots[:, 1] >= shape[0]*shape[2]).any():
        raise ValueError('Projected slot outside configured lattice.')
    y, u = slots.astype(np.int64).T
    x, z = np.divmod(u, shape[0])
    return np.column_stack((np.zeros(len(y), dtype=np.int64), z, y, x))


def inverse_project_xy(xy, shape, theta):
    """Match projected coordinates to the unique rotated voxel center (paper Fig. 8).

    Integer slots used by the stream are exact labels of these centers, so the
    normal decoder can invert labels without floating point search or binning.
    """
    shape = validate_shape(shape)
    theta = _angle(shape, theta)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Expected finite projected coordinates (N,2).')
    angle = np.deg2rad(theta)
    x, z = np.divmod(np.arange(shape[0]*shape[2]), shape[0])
    centers = x*np.cos(angle)+z*np.sin(angle)
    if len(centers) > 1 and not (np.diff(centers) > 0).all():
        raise ValueError('Rotation cannot be resolved at float64 precision.')
    pos = np.clip(np.searchsorted(centers, xy[:, 0]), 0, len(centers)-1)
    left = np.maximum(pos-1, 0)
    use_left = np.abs(centers[left]-xy[:, 0]) < np.abs(centers[pos]-xy[:, 0])
    pos = np.where(use_left, left, pos)
    tol = 32*np.finfo(np.float64).eps*max(float(centers[-1]), 1.)
    y = np.rint(xy[:, 1])
    if (np.abs(centers[pos]-xy[:, 0]) > tol).any() or (np.abs(y-xy[:, 1]) > tol).any():
        raise ValueError('Point does not match a configured rotated lattice center.')
    return lattice_inverse(np.column_stack((y, pos)), shape)


def _validate(coords, features, shape, q, theta):
    shape = validate_shape(shape)
    if coords.ndim != 2 or coords.shape[1] != 4 or features.ndim != 2 or len(coords) != len(features) or features.shape[1] < 1:
        raise ValueError('Expected aligned coordinates (N,4) and features (N,C).')
    if coords.dtype.kind not in 'iuf' or features.dtype.kind not in 'iuf' or not np.isfinite(coords).all() or (coords != np.floor(coords)).any() or not np.isfinite(features).all():
        raise ValueError('Finite integer coordinates and finite real attributes required.')
    if (coords[:, 0] != 0).any() or (coords[:, 1:] < 0).any() or (coords[:, 1:] >= np.array(shape)).any():
        raise ValueError('Single batch zero, in-bounds coordinates required.')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('Duplicate coordinates are ambiguous.')
    _angle(shape, theta)
    if not np.isfinite(q) or q <= 0:
        raise ValueError('Quantization multiplier must be finite and positive.')
    if shape[0]*shape[2]*features.shape[1] > 4_000_000:
        raise ValueError('Reference attribute patch exceeds four million elements.')
    return shape


def encode(coords, features, shape, prefix, q, theta=None):
    import deepCABAC
    coords, features = np.asarray(coords), np.asarray(features)
    theta = _angle(shape, theta)
    shape = _validate(coords, features, shape, q, theta)
    quantized = np.rint(features.astype(np.float64)*q)
    if np.abs(quantized).max(initial=0) > 2**22:
        raise ValueError('Quantized attributes exceed safe exact CABAC residual range.')
    slots, _ = lattice_projection(coords, shape, theta)
    order = np.lexsort((slots[:, 1], slots[:, 0]))
    slots, quantized = slots[order], quantized[order]
    ge, ae = deepCABAC.Encoder(), deepCABAC.Encoder()
    previous_geo = np.zeros((shape[2], shape[0]), dtype=bool)
    previous_attr = np.zeros((shape[0]*shape[2], features.shape[1]), dtype=np.float64)
    counts = []
    for y in range(shape[1]):
        selected = slots[:, 0] == y
        u = slots[selected, 1]
        occupied = np.zeros_like(previous_geo)
        occupied.reshape(-1)[u] = 1
        ge.encodeWeightsRD(np.logical_xor(occupied, previous_geo).astype(np.float32), 1., 1., 0.)
        values = quantized[selected]
        counts.append(len(values))
        if len(values):
            ae.encodeWeightsRD((values-previous_attr[u]).astype(np.float32), 1., 1., 0.)
        current_attr = np.zeros_like(previous_attr)
        current_attr[u] = values
        previous_attr, previous_geo = current_attr, occupied
    meta = {'version': 2, 'shape_source': 'decoder_config', 'channels': int(features.shape[1]),
            'q': float(q), 'theta_degrees': theta, 'patch_axis': 'y',
            'slot_order': 'x*Z+z', 'attribute_prediction': 'previous_patch_same_cell',
            'counts': counts, 'point_count': len(coords)}
    header = json.dumps(meta, separators=(',', ':')).encode()
    prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    geometry = MAGIC + struct.pack('<I', len(header)) + header + ge.finish().tobytes()
    attributes = ae.finish().tobytes() if len(coords) else b''
    Path(str(prefix)+'_geometry.bin').write_bytes(geometry)
    Path(str(prefix)+'_attributes.bin').write_bytes(attributes)
    return {'point_count': len(coords), 'geometry_bytes': len(geometry),
            'attribute_bytes': len(attributes), 'header_bytes': len(MAGIC)+4+len(header),
            'total_bytes': len(geometry)+len(attributes)}


def decode(prefix, shape=None):
    import deepCABAC
    prefix = Path(prefix)
    raw = Path(str(prefix)+'_geometry.bin').read_bytes()
    magic = raw[:len(MAGIC)]
    if magic not in {MAGIC, LEGACY_MAGIC} or len(raw) < len(MAGIC)+4:
        raise ValueError('Not a supported reference stream.')
    size = struct.unpack('<I', raw[len(MAGIC):len(MAGIC)+4])[0]
    end = len(MAGIC)+4+size
    if size > 1_000_000 or end >= len(raw):
        raise ValueError('Invalid geometry header.')
    meta = json.loads(raw[len(MAGIC)+4:end])
    version = 1 if magic == LEGACY_MAGIC else 2
    if meta['version'] != version:
        raise ValueError('Stream version mismatch.')
    if version == 1:
        if shape is not None and validate_shape(shape) != validate_shape(meta['shape_zyx']):
            raise ValueError('Configured shape differs from version 1 header.')
        shape = meta['shape_zyx']
    elif shape is None:
        raise ValueError('Version 2 decode requires the fixed --shape from model configuration.')
    elif meta.get('attribute_prediction') != 'previous_patch_same_cell':
        raise ValueError('Unsupported attribute predictor.')
    shape = validate_shape(shape)
    channels, q = meta['channels'], meta['q']
    if type(channels) is not int or channels < 1 or channels > 4096:
        raise ValueError('Invalid channel count.')
    _validate(np.empty((0,4)), np.empty((0,channels)), shape, q, meta['theta_degrees'])
    counts = meta['counts']
    if len(counts) != shape[1] or any(type(n) is not int or n < 0 or n > shape[0]*shape[2] for n in counts) or sum(counts) != meta['point_count']:
        raise ValueError('Inconsistent stream metadata.')
    gd = deepCABAC.Decoder(); gd.getStream(np.frombuffer(raw[end:], dtype=np.uint8))
    ar = Path(str(prefix)+'_attributes.bin').read_bytes()
    ad = deepCABAC.Decoder()
    if meta['point_count']:
        if not ar: raise ValueError('Missing attribute payload.')
        ad.getStream(np.frombuffer(ar, dtype=np.uint8))
    elif ar:
        raise ValueError('Unexpected attributes for empty frame.')
    previous_geo = np.zeros((shape[2], shape[0]), dtype=bool)
    previous_attr = np.zeros((shape[0]*shape[2], channels), dtype=np.float64)
    dc_predictor = np.zeros(channels)
    positions, values = [], []
    for y, count in enumerate(counts):
        delta = gd.decodeWeights()
        if delta.shape != previous_geo.shape or not np.isin(delta, [0,1]).all():
            raise ValueError('Invalid geometry delta.')
        occupied = np.logical_xor(previous_geo, delta)
        u = np.flatnonzero(occupied)
        if len(u) != count: raise ValueError('Geometry/attribute count mismatch.')
        current_attr = np.zeros_like(previous_attr)
        if count:
            residual = ad.decodeWeights()
            if residual.shape != (count,channels) or not np.isfinite(residual).all() or (residual != np.rint(residual)).any():
                raise ValueError('Invalid attribute residual.')
            prediction = dc_predictor if version == 1 else previous_attr[u]
            restored = residual.astype(np.float64)+prediction
            current_attr[u] = restored
            positions.append(np.column_stack((np.full(count,y),u)))
            values.append(restored/q)
            dc_predictor = np.rint(restored.mean(axis=0))
        else:
            dc_predictor = np.zeros(channels)
        previous_geo, previous_attr = occupied, current_attr
    slots = np.concatenate(positions) if positions else np.empty((0,2),dtype=np.int64)
    features = np.concatenate(values).astype(np.float32) if values else np.empty((0,channels),dtype=np.float32)
    # Integer lookup exactly identifies the rotated grid center and inverse rotation.
    coords = lattice_inverse(slots, shape).astype(np.int32)
    return coords, features, dict(meta, shape_zyx=list(shape))


class SparseCodec:
    """Explicit inference/export adapter; does not select a detector backend."""
    def __init__(self, stream_dir, q, theta=None, spatial_shape=None):
        self.stream_dir, self.q, self.theta = Path(stream_dir), q, theta
        self.spatial_shape = validate_shape(spatial_shape) if spatial_shape is not None else None

    def _prefix(self, frame):
        if Path(frame).name != frame or frame in {'.','..'}: raise ValueError('Invalid frame.')
        return self.stream_dir/frame

    def encode(self, tensor, frame):
        if tensor.batch_size != 1:
            raise ValueError('Single batch required.')
        if self.spatial_shape is not None and validate_shape(tensor.spatial_shape) != self.spatial_shape:
            raise ValueError('Tensor and configured decoder grids differ.')
        return encode(tensor.indices.detach().cpu().numpy(), tensor.features.detach().cpu().numpy(),
                      tensor.spatial_shape, self._prefix(frame), self.q, self.theta)

    def decode(self, frame, device):
        import torch
        import spconv.pytorch as spconv
        coords, features, meta = decode(self._prefix(frame), self.spatial_shape)
        return spconv.SparseConvTensor(torch.as_tensor(features,device=device),
                                      torch.as_tensor(coords,device=device),meta['shape_zyx'],1)
