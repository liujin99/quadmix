"""Shared memory helpers for multiprocessing (avoids re-reading large metadata per worker)."""

import multiprocessing as mp
import multiprocessing.shared_memory

import numpy as np


class SharedArrayInfo:
    """Descriptor for a numpy array in shared memory — pickle-safe for mp."""

    def __init__(self, name: str, shape: tuple, dtype: str, nbytes: int):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes


def ndarray_to_shared(arr: np.ndarray, prefix: str) -> SharedArrayInfo:
    """Copy numpy array into shared memory, return descriptor."""
    shm = mp.shared_memory.SharedMemory(create=True, size=arr.nbytes, name=f"{prefix}_shm")
    shared = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(shared, arr)
    info = SharedArrayInfo(name=shm.name, shape=arr.shape, dtype=str(arr.dtype), nbytes=arr.nbytes)
    shm.close()
    return info


def shared_to_ndarray(info: SharedArrayInfo) -> np.ndarray:
    """Map shared memory back to numpy array (zero-copy read-only view).

    Uses ctypes shm_open + mmap.mmap + os.close(fd) to attach. Python 3.13+
    SharedMemory.close() also munmaps _mmap, which would segfault the view.
    Here os.close(fd) closes the fd but the mmap object (held by the returned
    ndarray's base) retains the mapping until the ndarray is GC'd.
    """
    import ctypes, mmap, os
    libc = ctypes.CDLL('libc.so.6')
    fd = libc.shm_open(info.name.encode(), os.O_RDWR, 0o600)
    if fd < 0:
        raise OSError(f"shm_open failed for {info.name}: fd={fd}")
    m = mmap.mmap(fd, info.nbytes)
    os.close(fd)
    arr = np.ndarray(shape=info.shape, dtype=np.dtype(info.dtype), buffer=m)
    arr.flags.writeable = False
    return arr
