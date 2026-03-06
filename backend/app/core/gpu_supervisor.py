"""VRAM-safe GPU process supervisor with fault isolation.

Long-lived daemon processes pinned to specific CUDA devices.  Communication
uses ``multiprocessing.Queue`` for metadata; payloads pass via filesystem
to prevent ``PicklingError``.

Resolves **F5** (PyTorch thread zombies) and **F10** (hardware underutilization).
"""
from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import queue
import time
from typing import Any

logger = logging.getLogger(__name__)


def force_kill_process_tree(pid: int) -> None:
    """Recursively kill a process and all its children (F5 fix).

    PyTorch spawns C++ background threads (DataLoader workers, CUDA streams)
    that orphan themselves if only the parent is killed.  ``psutil`` walks
    the full process tree and kills each child before the parent.

    Falls back to ``os.kill`` if ``psutil`` is not available.
    """
    try:
        import psutil

        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        except psutil.NoSuchProcess:
            pass
    except ImportError:
        import os
        import platform
        import signal

        try:
            if platform.system() != "Windows":
                os.kill(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass


def _gpu_worker_loop(req_q: mp.Queue, res_q: mp.Queue, device_id: int) -> None:
    """Worker loop — runs in a child process with its own CUDA context.

    **STRICT INVARIANT**: ``torch`` is imported ONLY inside this child process
    to bind a clean CUDA context.

    Parameters
    ----------
    req_q : Queue
        Receives job dicts with ``plugin_path``, ``context``, ``optimize_ada``.
    res_q : Queue
        Sends result dicts back to the supervisor.
    device_id : int
        CUDA device ordinal to pin this worker to.
    """
    import torch  # noqa: F811 — intentionally delayed import

    torch.cuda.set_device(device_id)

    # VRAM cache survives across ticks — prevents NVMe reload latency
    model_cache: dict[str, Any] = {}

    while True:
        job = req_q.get()
        if job is None:
            break
        try:
            plugin = importlib.import_module(job["plugin_path"])

            # F10: Ada Lovelace Hardware Optimization with JIT Warmup
            if device_id == 1 and job.get("optimize_ada", False):
                torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(plugin, "model") and not getattr(
                    plugin, "_is_compiled", False
                ):
                    # Signal parent to suspend timeout clock during JIT compile
                    res_q.put({"status": "compiling"})
                    plugin.model = torch.compile(
                        plugin.model.to(torch.bfloat16), mode="reduce-overhead"
                    )
                    # Force JIT warmup pass to trigger compilation immediately
                    _ = plugin.model(
                        torch.randn(
                            1, 64, device=f"cuda:{device_id}", dtype=torch.bfloat16
                        )
                    )
                    plugin._is_compiled = True

            # Plugin reads context (JSON paths), runs inference, writes targets
            plugin.execute(job["context"], model_cache)
            res_q.put({"status": "success"})
        except Exception as e:
            res_q.put({"status": "error", "error": str(e)})


def optimize_model_for_ada(model: Any) -> Any:
    """Apply Ada Lovelace (RTX 4070 Super) hardware optimizations.

    1. BFloat16 prevents Gumbel-Softmax underflow common in RL/SMoE.
    2. Enable TF32 Tensor Cores for matmul.
    3. ``torch.compile`` with ``reduce-overhead`` fuses CUDA kernels,
       eliminating CPU-to-GPU PCIe memory bandwidth bottlenecks.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to optimize.

    Returns
    -------
    Compiled, bf16-cast model.
    """
    import torch

    model = model.to(dtype=torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    return torch.compile(model, mode="reduce-overhead")


class GPUProcessSupervisor:
    """Manages a fault-isolated GPU worker process.

    On timeout, the worker and all its child threads are recursively killed
    via ``psutil`` (VRAM freed immediately) and a fresh worker is spawned.

    Uses ``mp.get_context('spawn')`` to guarantee clean memory boundaries.

    Parameters
    ----------
    device_id : int
        CUDA device ordinal (e.g. ``0`` for training, ``1`` for inference).
    timeout_s : int
        Maximum seconds to wait for a job result.
    """

    def __init__(self, device_id: int, timeout_s: int = 120) -> None:
        self.device_id = device_id
        self.timeout_s = timeout_s
        self.ctx = mp.get_context("spawn")
        self._spawn_worker()

    def _spawn_worker(self) -> None:
        """Start a fresh GPU worker process.

        Queues are recreated on each respawn to avoid OS pipe deadlocks.
        """
        self.req_q: mp.Queue = self.ctx.Queue()
        self.res_q: mp.Queue = self.ctx.Queue()
        self.process = self.ctx.Process(
            target=_gpu_worker_loop,
            args=(self.req_q, self.res_q, self.device_id),
            daemon=True,
        )
        self.process.start()
        logger.info(
            "GPU worker spawned: device=%d pid=%d", self.device_id, self.process.pid
        )

    def execute_job(
        self,
        plugin_path: str,
        context: dict[str, Any],
        optimize_ada: bool = False,
    ) -> dict[str, Any]:
        """Submit a job to the GPU worker and wait for result.

        If the worker signals ``{"status": "compiling"}``, the timeout
        clock is reset to allow JIT compilation to complete without
        triggering a false timeout kill.

        Parameters
        ----------
        plugin_path : str
            Dotted module path (e.g. ``backend.app.sleeves.mera_plugin``).
        context : dict
            JSON-serializable context dict (artifact paths, asof_date, etc.).
        optimize_ada : bool
            Whether to apply Ada Lovelace TF32/compile optimizations.

        Returns
        -------
        dict with ``status``.

        Raises
        ------
        TimeoutError
            If the worker exceeds ``timeout_s`` (excluding JIT compile time).
        RuntimeError
            If the worker reports an error.
        """
        self.req_q.put({
            "plugin_path": plugin_path,
            "context": context,
            "optimize_ada": optimize_ada,
        })
        start_time = time.monotonic()

        while True:
            if (time.monotonic() - start_time) > self.timeout_s:
                logger.error(
                    "GPU worker %d (pid=%d) timed out after %ds — killing tree",
                    self.device_id,
                    self.process.pid,
                    self.timeout_s,
                )
                force_kill_process_tree(self.process.pid)
                self.process.join(timeout=5)
                self._spawn_worker()
                raise TimeoutError(
                    f"Worker {self.device_id} timed out. "
                    "Process tree killed. VRAM reclaimed."
                )

            try:
                res = self.res_q.get(timeout=1.0)
                if res["status"] == "compiling":
                    # Reset timeout clock — JIT warmup can take 2-5 minutes
                    start_time = time.monotonic()
                    logger.info(
                        "GPU worker %d: JIT compiling — timeout clock reset",
                        self.device_id,
                    )
                    continue
                if res["status"] == "error":
                    raise RuntimeError(res["error"])
                return res
            except queue.Empty:
                continue
