"""Fixed decoder geometry derived from the retained VirConv grid configuration."""
import numpy as np


def validate_shape(shape):
    value = np.asarray(shape)
    if value.shape != (3,) or value.dtype.kind not in 'iuf' or not np.isfinite(value).all() or (value <= 0).any() or (value != np.floor(value)).any():
        raise ValueError('Expected three positive integer spatial dimensions [Z,Y,X].')
    return tuple(int(v) for v in value)


def feature_shapes(input_shape):
    """Apply the actual stride-2 kernel-3 convolution shape rules, not occupancy maxima."""
    shape = np.array(validate_shape(input_shape), dtype=np.int64)
    f2 = (shape + 1) // 2  # padding (1,1,1), stride 2
    f3 = (f2 + 1) // 2
    f4 = (f3 + np.array([0, 2, 2]) - 3) // 2 + 1  # padding (0,1,1)
    return {'F3': validate_shape(f3), 'F4': validate_shape(f4)}
