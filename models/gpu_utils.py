import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

# CuPy Block Parameters
BLOCKSIZE = 1024
BLOCKNUM = lambda x : (x - 1) // BLOCKSIZE + 1

# Conversion between Torch and CuPy
def cupy_to_torch(x): return from_dlpack(x.toDlpack())
def torch_to_cupy(x): return cp.fromDlpack(to_dlpack(x))
