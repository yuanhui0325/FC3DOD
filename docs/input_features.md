# Input features

PENet generates eight point attributes before voxelization:

| Zero-based column | Attribute |
|---|---|
| 0, 1, 2 | x, y, z |
| 3 | LiDAR intensity, scaled by 10 during preprocessing; virtual points initially use zero |
| 4, 5, 6 | RGB, scaled by 1/3 during preprocessing; the detection backbone zeros these columns |
| 7 | Point source: 1 for virtual points, 2 for real LiDAR points |

`MeanVFE` averages point attributes. With `MODEL: max`, the last column uses the maximum source flag: mixed real/virtual voxels receive 2, and virtual-only voxels receive 1.

Sparse coordinates are stored separately as `[batch, z, y, x]`. Run the supplied configurations with batch size 1, batch index 0 and a single view.

## Feature transfer

During training, floating-point branch features pass directly to the cloud modules. During evaluation, the model applies G-PCC geometry coding and CABAC attribute coding. Use `model.eval()` together with `torch.no_grad()` for evaluation.

Both encoder and decoder use the grid derived from the model configuration. The application attribute header does not transmit spatial dimensions.
