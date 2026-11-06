"""Real CABAC attribute payload, with grid dimensions supplied out of band."""
import numpy as np
from .codec_config import validate_shape


def encode_quantized(values, interval=1., step_size=1., lambda_rd=0.):
    import deepCABAC
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite quantized attributes (N,C).')
    encoder = deepCABAC.Encoder()
    encoder.encodeWeightsRD(values, interval, step_size, lambda_rd)
    return encoder.finish().tobytes()


def decode_quantized(payload, spatial_shape, legacy_shape_header=False):
    import deepCABAC
    shape = validate_shape(spatial_shape)
    if legacy_shape_header:
        if len(payload) <= 3 or tuple(payload[:3]) != tuple(v % 256 for v in shape):
            raise ValueError('Legacy shape header does not match the configured decoder shape.')
        payload = payload[3:]
    if not payload:
        raise ValueError('Empty CABAC attribute payload.')
    decoder = deepCABAC.Decoder()
    decoder.getStream(np.frombuffer(payload, dtype=np.uint8))
    values = decoder.decodeWeights()
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Invalid CABAC attributes.')
    return values
