import os
import numpy as np
import h5py


def read_h5_geo(filedir):
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:,0:3].astype('int')

    return coords

def write_h5_geo(filedir, coords):
    data = coords.astype('uint8')
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)

    return

def read_ply_ascii_geo(filedir):
    """Read the first three vertex properties from an ASCII PLY, including zero vertices."""
    with open(filedir) as f:
        if f.readline().strip() != 'ply':
            raise ValueError('Expected PLY header')
        count = None
        ascii_format = False
        for line in f:
            words = line.split()
            if words[:2] == ['format', 'ascii']:
                ascii_format = True
            if words[:2] == ['element', 'vertex']:
                count = int(words[2])
            if line.strip() == 'end_header':
                break
        else:
            raise ValueError('Missing end_header')
        if not ascii_format or count is None:
            raise ValueError('Expected ASCII PLY with vertex count')
        data = []
        for _ in range(count):
            values = f.readline().split()
            if len(values) < 3:
                raise ValueError('Truncated PLY vertex data')
            data.append([float(v) for v in values[:3]])
    return np.asarray(data).reshape(-1, 3).astype(int)


def write_ply_ascii_geo(filedir, coords):
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError('Expected finite coordinates of shape (N, 3)')
    with open(filedir, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(coords))
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        for p in coords.astype(int):
            f.write('%d %d %d\n' % tuple(p))


import torch

def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().cpu(), step.long().cpu()


    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    device = data.device
    data, ground_truth = data.cpu(), ground_truth.cpu()
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())

    return torch.Tensor(mask).bool().to(device)

def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)

def sort_spare_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    indices_sort = np.argsort(array2vector(sparse_tensor.indices.cpu(),
                                           sparse_tensor.indices.cpu().max()+1))
    indices_old = sparse_tensor.indices
    feature_old = sparse_tensor.features
    indices_new = indices_old[indices_sort]
    sparse_tensor.indices = indices_new
    feature_new = feature_old[indices_sort]
    sparse_tensor = sparse_tensor.replace_feature(feature_new)
    sparse_tensor_sort = sparse_tensor


    return sparse_tensor_sort

def load_sparse_tensor(filedir, device):
    import MinkowskiEngine as ME
    coords = torch.tensor(read_ply_ascii_geo(filedir)).int()
    feats = torch.ones((len(coords),1)).float()

    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
    
    return x

def scale_sparse_tensor(x, factor):
    import MinkowskiEngine as ME
    coords = (x.C[:,1:]*factor).round().int()
    feats = torch.ones((len(coords),1)).float()
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)
    
    return x
