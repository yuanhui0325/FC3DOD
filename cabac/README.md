# deepCABAC Python extension

This directory retains the deepCABAC C++ implementation and pybind11 bindings used by FC3DOD. See [LICENSE](LICENSE) and the source copyright notices for the component's terms. The detector's overall setup does not override this license.

## Build

Install the build dependencies from the FC3DOD root, then build this extension:

```bash
python -m pip install -r requirements-build.txt
(cd cabac && python -m pip install --no-build-isolation --no-deps .)
```

A working C++ compiler and compatible Python/pybind11 installation are required. The existing component-test environment has deepCABAC 0.1.0; a clean build has not been validated on every platform.

## Real roundtrip example

```python
import numpy as np
import deepCABAC

attributes = np.array([[1, -2], [0, 3]], dtype=np.float32)
encoder = deepCABAC.Encoder()
encoder.encodeWeightsRD(attributes, 1.0, 1.0, 0.0)
stream = encoder.finish()

decoder = deepCABAC.Decoder()
decoder.getStream(stream)
restored = decoder.decodeWeights()
assert np.array_equal(attributes, restored)
```

These unit-step integer values have a tested exact roundtrip. This does not imply lossless encoding for arbitrary parameters or floating-point attributes. FC3DOD applies its explicit attribute quantizer before coding.

For project codec tools and tests, see the [main README](../README.md), [projected codec](../docs/projected_codec.md) and `tests/verify_components.py`.
