from __future__ import annotations

import ctypes
import ctypes.util
import threading


class ActiveAllocationTracker:
    def __init__(self) -> None:
        self._active: dict[int, int] = {}
        self.active_bytes = 0
        self.peak_active_bytes = 0

    def allocate(self, pointer: int, size: int) -> None:
        if pointer in self._active:
            raise ValueError(f"allocation pointer {pointer} is already active")
        self._active[pointer] = size
        self.active_bytes += size
        self.peak_active_bytes = max(self.peak_active_bytes, self.active_bytes)

    def free(self, pointer: int) -> None:
        self.active_bytes -= self._active.pop(pointer)

    def reset_peak(self) -> None:
        self.peak_active_bytes = self.active_bytes


class CudaAllocationTrace:
    def __init__(self) -> None:
        candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
        ]
        error: OSError | None = None
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                self._cudart = ctypes.CDLL(candidate)
                break
            except OSError as caught:
                error = caught
        else:
            raise RuntimeError("could not load CUDA runtime") from error

        self._cudart.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self._cudart.cudaMalloc.restype = ctypes.c_int
        self._cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self._cudart.cudaFree.restype = ctypes.c_int
        self._lock = threading.Lock()
        self._active = ActiveAllocationTracker()
        self.allocations: list[int] = []
        self.errors: list[str] = []

        alloc_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
        free_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        self.alloc_callback = alloc_type(self._allocate)
        self.free_callback = free_type(self._free)

    @property
    def alloc_address(self) -> int:
        return ctypes.cast(self.alloc_callback, ctypes.c_void_p).value or 0

    @property
    def free_address(self) -> int:
        return ctypes.cast(self.free_callback, ctypes.c_void_p).value or 0

    @property
    def active_bytes(self) -> int:
        with self._lock:
            return self._active.active_bytes

    @property
    def peak_active_bytes(self) -> int:
        with self._lock:
            return self._active.peak_active_bytes

    def begin_measurement(self) -> None:
        with self._lock:
            self.allocations.clear()
            self._active.reset_peak()

    def _allocate(self, size: int) -> int | None:
        pointer = ctypes.c_void_p()
        result = self._cudart.cudaMalloc(ctypes.byref(pointer), size)
        if result != 0 or pointer.value is None:
            with self._lock:
                self.errors.append(
                    f"cudaMalloc({size}) failed with CUDA error {result}"
                )
            return None
        with self._lock:
            self.allocations.append(size)
            self._active.allocate(pointer.value, size)
        return pointer.value

    def _free(self, pointer: int | None) -> None:
        if pointer is None:
            return
        result = self._cudart.cudaFree(ctypes.c_void_p(pointer))
        with self._lock:
            try:
                self._active.free(pointer)
            except KeyError:
                self.errors.append(f"cudaFree({pointer}) did not match an allocation")
            if result != 0:
                self.errors.append(
                    f"cudaFree({pointer}) failed with CUDA error {result}"
                )
