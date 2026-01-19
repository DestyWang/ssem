import cupy as cp
import torch
from time import time
# from .gpu_utils import BLOCKSIZE, BLOCKNUM, cupy_to_torch, torch_to_cupy
from gpu_utils import BLOCKSIZE, BLOCKNUM, cupy_to_torch, torch_to_cupy
import mrcfile
__all__ = ['Wasserstein2Loss']

ker_blur = cp.RawKernel(r'''
extern "C" __global__ void blur(
    const float* src,
    int size,
    int n,
    float B,
    float* dst)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size) {
        int offset = tid % n;
        int base = tid - offset;
        float w = src[tid];
        for (int i = 0; i < n; ++i)
            w = fmaxf(w, src[base + i] - (i - offset) * (i - offset) / B);
        float v = 0;
        for (int i = 0; i < n; ++i)
            v += expf(src[base + i] - (i - offset) * (i - offset) / B - w);
        dst[tid] = logf(v) + w;
    }
}
''', 'blur')

def blur(b, x, start_axis: int = 0):
    """
    Apply the log-sum-exp quadratic blur (Sinkhorn convolution) along axes [start_axis, ..., x.ndim-1].

    Notes
    -----
    - This function assumes `x` is a CuPy array of dtype float32.
    - Axes before `start_axis` are treated as batch axes and will NOT be blurred.
    """
    if x.ndim <= start_axis:
        return x
    # We reuse a single temporary buffer with the same total size.
    tmp = cp.empty_like(x, dtype=cp.float32)
    for axis in range(start_axis, x.ndim):
        # Move the selected axis to the last dimension.
        x_m = cp.moveaxis(x, axis, -1).copy()
        tmp_m = tmp.reshape(x_m.shape)

        # Blur along last axis (length = n).
        ker_blur(
            (BLOCKNUM(x_m.size),),
            (BLOCKSIZE,),
            (x_m, x_m.size, x_m.shape[-1], cp.float32(b), tmp_m),
        )

        # Move the last axis back to its original position.
        x = cp.moveaxis(tmp_m, -1, axis).copy()
    return x

class Wasserstein2LossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a0, a1, paras):
        assert torch.is_tensor(a0) and torch.is_tensor(a1)
        assert a0.ndim >= 2 and a1.ndim >= 2, "Expected a0/a1 shapes: (B, n1, ..., nk) with k>=1."
        assert a0.shape[1:] == a1.shape[1:], "Spatial shapes (n1..nk) must match."
        assert a0.is_cuda and a1.is_cuda and a0.device == a1.device

        eps = paras.get('eps', 1e-3)
        delta = paras.get('delta', 1e-3)
        maxiter = paras.get('maxiter', 1000)
        verbose = paras.get('verbose', 0)
        batch = paras.get('batch', 200)
        # only use spatial dimensions to define pixel size
        spatial_shape = a0.shape[1:]
        pixel_size = 1 / max(spatial_shape)
        if verbose >= 1: print('Perform Sinkhorn algorithm for solving modified Kantorovich potentials.')

        if verbose >= 1: print('  Initializing.')
        B0 = int(a0.shape[0])
        B1 = int(a1.shape[0])

        a0 = torch_to_cupy(a0)  # (B0, n1, ..., nk)
        a1 = torch_to_cupy(a1)  # (B1, n1, ..., nk)
        with a0.device:
            b = eps / (pixel_size * pixel_size)
            # Build all pair distributions (B0,B1, n1,...,nk) via broadcasting.
            # This is required because the output is a full pairwise matrix (B0 x B1).
            full_shape = (B0, B1) + tuple(spatial_shape)
            a0_full = cp.broadcast_to(a0[:, None, ...], full_shape)
            a1_full = cp.broadcast_to(a1[None, :, ...], full_shape)

            la0 = cp.log(cp.fmax(a0_full, 1e-20))
            la1 = cp.log(cp.fmax(a1_full, 1e-20))

            phi0 = cp.zeros(full_shape, dtype=cp.float32)
            phi1 = cp.zeros(full_shape, dtype=cp.float32)

            spatial_axes = tuple(range(2, phi0.ndim))
            # per-pair normalization (avoid mixing pairs in the stopping criterion)
            na0 = cp.sum(cp.abs(a0_full), axis=spatial_axes) + cp.float32(1e-20)

            if verbose >= 1: print('  Doing iteration.')
            time1 = time()
            for rd in range(maxiter):
                phi0 = la0 - blur(b, phi1, start_axis=2)
                phi1 = la1 - blur(b, phi0, start_axis=2)
                if rd == 0 or (rd + 1) % batch == 0:
                    # Compute per-pair marginal error, then use max over all pairs for stopping.
                    p1 = cp.exp(phi0 + blur(b, phi1, start_axis=2))
                    err = cp.sum(cp.abs(a0_full - p1), axis=spatial_axes) / na0  # (B0,B1)
                    err_max = float(err.max())
                    if verbose >= 2:
                        err_mean = float(err.mean())
                        print(f'    Round {rd + 1}, max(|i0 - P1|_1/|i0|_1) = {err_max:.6f}, mean = {err_mean:.6f}.')
                    if err_max < delta:
                        break
            time2 = time()
            if verbose >= 1: print(f'  Completed in {time2 - time1:.3f}s.')

            # Pairwise loss: output shape (B0,B1)
            p = cp.exp(phi0 + blur(b, phi1, start_axis=2))
            loss = -eps * cp.sum(p, axis=spatial_axes)  # (B0,B1)
            phi0 *= eps
            phi1 *= eps
            loss += cp.sum(phi0 * a0_full, axis=spatial_axes) + cp.sum(phi1 * a1_full, axis=spatial_axes)

        # Save for backward: we keep phi0 (as requested) and shapes for proper aggregation.
        ctx.phi0 = cupy_to_torch(phi0)  # (B0,B1,n1,...,nk)
        ctx.B0 = B0
        ctx.B1 = B1
        ctx.spatial_ndim = len(spatial_shape)
        return cupy_to_torch(loss)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (B0,B1)
        We return gradients w.r.t. a0: (B0,n1,...,nk), aggregated over the second batch dimension.

        NOTE: To keep behavior close to the original code, we only return grad for a0 (phi0),
        and return None for a1.
        """
        phi0 = ctx.phi0  # (B0,B1, n1,...,nk)
        B0 = ctx.B0
        B1 = ctx.B1
        k = ctx.spatial_ndim

        if grad_output is None:
            return None, None, None

        # Ensure grad_output shape is (B0,B1) for broadcasting.
        g = grad_output
        if g.ndim == 0:
            # scalar upstream grad (rare for matrix output); treat as all-ones scaling
            g = g.view(1, 1).expand(B0, B1)
        elif g.shape != (B0, B1):
            # Let torch attempt broadcasting, but we reshape to the expected batch dims if possible.
            g = g.reshape(B0, B1)

        # Broadcast g to (B0,B1,1,...,1) then aggregate over B1 -> (B0,n1,...,nk)
        view_shape = (B0, B1) + (1,) * k
        grad_a0 = (g.view(view_shape) * phi0).sum(dim=1)
        return grad_a0, None, None
    # def backward(ctx, grad_output):
    #     return grad_output * ctx.phi0, grad_output * ctx.phi1, None
    # torch.zeros_like(ctx.phi1, device=ctx.phi1.device)

class Wasserstein2Loss(torch.nn.Module):

    def __init__(self, **paras):
        super().__init__()
        self.paras = paras

    def forward(self, a, b):
        return Wasserstein2LossFunction.apply(a, b, self.paras)

if __name__ == '__main__':
    import numpy as np
    import os
    from skimage.io import imread
    from imageio import mimwrite
    device = torch.device('cuda')
    path0 = '/ShareX/wanghongyu/cryotrans/docking/data/pdb6rah_2A.mrc'
    with mrcfile.mmap(path0, permissive=True, mode='r') as mrc:
        map0 = np.array(mrc.data, dtype=np.float32, copy=True)
        map0 = torch.from_numpy(map0).to(device)

    path1 = '/ShareX/wanghongyu/cryotrans/docking/data/pdb6rah_lp.mrc'
    with mrcfile.mmap(path1, permissive=True, mode='r') as mrc:
        map1 = np.array(mrc.data, dtype=np.float32, copy=True)
        map1 = torch.from_numpy(map1).to(device)

    path2 = '/ShareX/wanghongyu/cryotrans/docking/data/pdb6rai_input_gmm.mrc'
    with mrcfile.mmap(path2, permissive=True, mode='r') as mrc:
        map2 = np.array(mrc.data, dtype=np.float32, copy=True)
        map2 = torch.from_numpy(map2).to(device)



    w2 = Wasserstein2Loss(eps = 1e-10, maxiter = 10)

    # map1[map1 < 0.02] = 0
    # map0[map0 < 0.24] = 0


    print('W2')
    map1 *= map0.sum() / map1.sum()

    print(w2(map0, map1))
    map1 *= map2.sum() / map1.sum()

    print(w2(map2, map1))