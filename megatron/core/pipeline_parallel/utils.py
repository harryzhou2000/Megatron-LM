# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Optional

import torch
from torch.autograd import Variable

from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank, make_viewless_tensor

logger = logging.getLogger(__name__)


def is_pp_first_stage(pp_group: torch.distributed.ProcessGroup):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pg_rank(pp_group) == 0


def is_pp_last_stage(pp_group: torch.distributed.ProcessGroup):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    return get_pg_rank(pp_group) == (get_pg_size(pp_group) - 1)


def is_vp_first_stage(vp_stage: int, vp_size: int | None):
    """Return True if in the first virtual pipeline model-parallel stage, False otherwise."""
    if vp_size is None or vp_size <= 1:
        assert vp_stage is None or vp_stage == 0, (
            f"Expected vp_stage to be 0 or None when vp_size is <= 1 or None, "
            f"but got vp_stage={vp_stage} and vp_size={vp_size}"
        )
        return True
    return vp_stage == 0


def is_vp_last_stage(vp_stage: int, vp_size: int | None):
    """Return True if in the last virtual pipeline model-parallel stage, False otherwise."""
    if vp_size is None or vp_size <= 1:
        assert vp_stage is None or vp_stage == 0, (
            f"Expected vp_stage to be 0 or None when vp_size is <= 1 or None, "
            f"but got vp_stage={vp_stage} and vp_size={vp_size}"
        )
        return True
    return vp_stage == (vp_size - 1)


def get_pp_first_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the first rank in the pipeline parallel group."""
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[0]


def get_pp_last_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the last rank in the pipeline parallel group."""
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[-1]


def get_pp_next_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the next rank in the pipeline parallel group, or None if last
    stage."""
    if is_pp_last_stage(pp_group):
        return None
    current_rank_in_group = get_pg_rank(pp_group)
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[current_rank_in_group + 1]


def get_pp_prev_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the previous rank in the pipeline parallel group, or None if
    first stage."""
    if is_pp_first_stage(pp_group):
        return None
    current_rank_in_group = get_pg_rank(pp_group)
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[current_rank_in_group - 1]


def make_viewless(e):
    """Make_viewless util func"""
    e = make_viewless_tensor(inp=e, requires_grad=e.requires_grad, keep_graph=True)
    return e


def set_ideal_affinity_for_current_gpu():
    """Set CPU affinity for the current GPU to optimize host-device transfers."""
    import uuid

    try:
        import cuda.bindings.driver as cuda_driver
        import cuda.bindings.runtime as cuda_runtime
    except:
        try:
            import cuda.cuda as cuda_driver
            import cuda.cudart as cuda_runtime
        except:
            raise RuntimeError("Please install cuda-python to enable GPU affinity setting")
    import pynvml

    # Get current CUDA device ID
    err, device_id = cuda_runtime.cudaGetDevice()
    assert err == cuda_runtime.cudaError_t.cudaSuccess
    # Get device UUID
    err, device_uuid = cuda_driver.cuDeviceGetUuid(device_id)
    assert err == cuda_driver.CUresult.CUDA_SUCCESS
    # Set CPU affinity based on GPU's NUMA node
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)

    log_single_rank(
        logger,
        logging.WARNING,
        f"Set CPU affinity for all GPUs for optimal host-device transfer performance",
    )


class NoopScheduleNode:
    """A placeholder node in the computation graph that simply passes through inputs and outputs.

    This class is used as a no-op node in the scheduling system when a real computation node
    is not needed but the interface must be maintained (e.g., dense layer doesn't need
    moe_dispatch and moe_combine). It simply returns its inputs unchanged
    in both forward and backward passes.
    """

    def forward(self, inputs):
        """Passes through inputs unchanged in the forward pass."""
        return inputs

    def backward(self, outgrads):
        """Passes through gradients unchanged in the backward pass."""
        return outgrads


class ScheduleNode:
    """Base node for fine-grained scheduling.

    This class represents a computational node in the pipeline schedule.
    It handles the execution of forward and backward operations on a stream.
    """

    def __init__(
        self,
        forward_func: Callable,
        stream: torch.cuda.Stream,
        event: torch.cuda.Event,
        backward_func: Optional[Callable] = None,
        free_input: bool = False,
        name: str = "schedule_node",
    ):
        """Initialize a schedule node.

        Args:
            forward_func (callable): Function to execute during the forward pass.
            stream (Callable): Func that returns CUDA stream for computation.
                This can be either a 'compute' stream or a 'communicate' stream.
                - 'compute' stream: Used for computational nodes like attention and experts.
                - 'communicate' stream: Used for nodes that handle token communication,
                  such as token dispatch and combine operations in MoE layers.
            event (torch.cuda.Event): The CUDA event used for synchronization. Each
                microbatch within a model chunk shares the same event, which is used
                to manage dependencies between nodes operating on different streams.
            backward_func (callable, optional): Function for the backward pass.
            free_input (bool): Flag to indicate if the input should be freed after the
                forward pass.
            name (str): Name of the node for debugging purposes.
        """
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func if backward_func else self.default_backward_func
        self.stream = stream
        self.event = event
        self.free_input = free_input
        self.inputs = None
        self.outputs = None
        self.delay_grads_release = False
        self.manual_release_grads = False
        # Pool-owned input buffers that must be released after backward.
        # Only populated for free_input=False nodes whose inputs came from
        # the ActivationPool.
        self._pool_inputs = None

    def default_backward_func(self, outputs, output_grad):
        """Default backward function"""
        Variable._execution_engine.run_backward(
            tensors=outputs,
            grad_tensors=output_grad,
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )
        return output_grad

    def forward(self, inputs=()):
        """Schedule node forward"""
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        # Lazy initialization of stream
        if isinstance(self.stream, Callable):
            self.stream = self.stream()

        pool = get_activation_pool()
        pool_active = pool is not None and pool.is_enabled

        with self.stream_acquire_context(f"{self.name} forward"):
            self.inputs = [make_viewless(e).detach() if e is not None else None for e in inputs]
            for i, input in enumerate(self.inputs):
                if input is not None:
                    input.requires_grad = inputs[i].requires_grad

            data = tuple(self.inputs)
            data = self.forward_func(*data)

            if not isinstance(data, tuple):
                data = make_viewless(data)
            else:
                data = tuple([make_viewless(e) if isinstance(e, torch.Tensor) else e for e in data])

            self.output = data

            # --- Pool-managed buffer reuse (must stay on self.stream) ------
            #
            # free_input=True nodes: release pool-owned input buffers now
            #     (the input data is no longer needed after forward).
            # free_input=False nodes: the input data is still needed for
            #     backward (self.inputs shares storage), so defer the
            #     release until _release_state() after backward completes.
            if pool_active:
                if self.free_input:
                    for input in inputs:
                        if input is not None:
                            pool.release(input)
                else:
                    # Remember pool-owned inputs for deferred release.
                    self._pool_inputs = [
                        input for input in inputs
                        if input is not None and pool.owns(input)
                    ]
            elif self.free_input:
                for input in inputs:
                    if input is not None:
                        input.record_stream(self.stream)
                        input.untyped_storage().resize_(0)

            if pool_active and self.free_input:
                pool_output = self._copy_output_to_pool(pool, self.output)
                # Free the original forward output storage — the data has
                # been copied into pool buffers.  self.output retains its
                # grad_fn graph for backward, but the underlying tensor
                # data is no longer needed (the pool copy is what flows to
                # the next node).
                self._free_output_storage(self.output)
                return pool_output

        return self.output

    def _copy_output_to_pool(self, pool, output):
        """Copy forward output data into pool-managed static buffers.

        The pool buffer is what flows to the *next* node (which detaches it
        anyway). ``self.output`` still points to the original tensor with
        its ``grad_fn``, so this node's backward is unaffected.
        """
        if isinstance(output, tuple):
            return tuple(
                self._copy_single_to_pool(pool, t) if isinstance(t, torch.Tensor) else t
                for t in output
            )
        if isinstance(output, torch.Tensor):
            return self._copy_single_to_pool(pool, output)
        return output

    @staticmethod
    def _copy_single_to_pool(pool, tensor):
        """Acquire a buffer from the pool and copy tensor data into it."""
        buf = pool.acquire(tuple(tensor.shape), tensor.dtype)
        if buf is None:
            return tensor
        # Use no_grad to avoid autograd tracking on the pool buffer.
        # The next node detaches its input anyway (line 205), so the pool
        # buffer never participates in the autograd graph.
        with torch.no_grad():
            buf.copy_(tensor)
            buf.requires_grad_(tensor.requires_grad)
        return buf

    @staticmethod
    def _free_output_storage(output):
        """Free the underlying storage of forward output tensors.

        After the data has been copied into pool buffers, the original
        tensors' storage is dead weight — backward only needs the
        ``grad_fn`` graph, not the data.  Releasing storage here mirrors
        what the non-pool path does for inputs (``resize_(0)``).
        """
        if isinstance(output, tuple):
            for t in output:
                if isinstance(t, torch.Tensor):
                    t.untyped_storage().resize_(0)
        elif isinstance(output, torch.Tensor):
            output.untyped_storage().resize_(0)

    def get_output(self):
        """Get the forward output"""
        return self.output

    def backward(self, output_grad):
        """Schedule node backward"""
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad)

    def _backward(self, *output_grad):
        # Lazy initialization of stream
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} backward"):
            outputs = self.output
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            assert len(outputs) == len(output_grad), (
                f"{len(outputs)} of {type(outputs[0])} is not equal to "
                f"{len(output_grad)} of {type(output_grad[0])}"
            )
            output_grad = self.backward_func(outputs, output_grad)

        # output_grad maybe from another stream
        if output_grad:
            for g in output_grad:
                if g is not None:
                    g.record_stream(self.stream)
                    # Manually trigger the memory release of dgrad tensor
                    # to avoid delayed garbage collection. If
                    # delay_grads_release is True, dgrad is last used in
                    # wgrad compute and skip the release here.
                    if self.manual_release_grads and not self.delay_grads_release:
                        g.untyped_storage().resize_(0)

        grads = self.get_grad()
        self._release_state()

        return grads

    def get_grad(self):
        """Get the grad of inputs"""
        grad = tuple([e.grad if e is not None else None for e in self.inputs])
        # multiple in, multiple out
        if len(grad) == 1:
            grad = grad[0]
        return grad

    @contextmanager
    def stream_acquire_context(self, name=None):
        """Stream acquire context that handles event synchronization,
            NVTX profiling, and stream context.

        This context manager consolidates:
        1. Event wait/record for synchronization between streams
        2. NVTX range for profiling (if name is provided)
        3. torch.cuda.stream context for execution on the specified stream

        Args:
            name: Optional name for NVTX range profiling
        """
        self.event.wait(self.stream)
        if name:
            torch.cuda.nvtx.range_push(name)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            if name:
                torch.cuda.nvtx.range_pop()
            self.event.record(self.stream)

    def _release_state(self):
        """Clear the state of the node"""
        # Release pool-owned input buffers that were deferred from forward
        # (free_input=False nodes whose inputs came from the ActivationPool).
        if self._pool_inputs:
            pool = get_activation_pool()
            if pool is not None:
                for t in self._pool_inputs:
                    pool.release(t)
            self._pool_inputs = None
        self.inputs = None
        self.output = None
        del self.forward_func
        del self.backward_func


class AbstractSchedulePlan(ABC):
    """To use combined 1f1b, model must implement build_schedule_plan while take the same
    signature as model forward but return an instance of AbstractSchedulePlan"""

    @staticmethod
    @abstractmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """run() is the protocol between our schedule logic and model, which is used to schedule
        the forward and backward schedule plans for the models.
        """
        ...


_USE_DYNAMIC_COMP_STREAM = None
_COMP_STREAM = None
_COMM_STREAM = None


def set_streams(comm_stream=None):
    """Set the stream for communication operations."""
    global _COMM_STREAM

    # Set communication stream
    if _COMM_STREAM is None:
        if comm_stream is None:
            comm_stream = torch.cuda.Stream(device="cuda")
        _COMM_STREAM = comm_stream


def get_comp_stream():
    """Get the stream for computation"""
    return torch.cuda.current_stream()


def get_comm_stream():
    """Get the stream for communication"""
    global _COMM_STREAM
    return _COMM_STREAM


# ---------------------------------------------------------------------------
# ActivationPool — static buffer reuse for combined_1f1b schedules.
#
# Problem: combined_1f1b processes M microbatches through L layers, each with
# free_input nodes that do record_stream() + resize_(0).  Even without CUDA
# graph this causes cudaMalloc/cudaFree churn.  Under full-iteration CUDA
# graph capture every allocation is additionally pinned in the graph's private
# pool and record_stream() causes massive fragmentation in the main allocator.
#
# Solution: Pre-allocate activation buffers that cycle between microbatches.
# Nodes copy their outputs into pool-owned buffers before returning to the
# next node.  Pool buffers are never freed — they are released back to the
# pool for the next microbatch to reuse at the same CUDA address.
# ---------------------------------------------------------------------------

_ACTIVATION_POOL = None


class ActivationPool:
    """Activation buffer pool for 1f1b inter-node tensor reuse.

    Manages a set of pre-allocated buffers keyed by ``(shape, dtype)``.  Each
    key has a small number of buffer slots (grows on demand during warmup).
    Buffers cycle through ``ACQUIRED`` / ``FREE`` states with strict ownership
    tracking to catch lifetime bugs early.

    Benefits:
        * Eliminates ``cudaMalloc`` / ``cudaFree`` churn from the
          ``free_input`` + ``resize_(0)`` pattern in ``ScheduleNode``.
        * Under full-iteration CUDA graph capture, keeps the graph's private
          pool small (2 microbatch slots instead of M) and avoids
          ``record_stream`` fragmentation in the main caching allocator.

    Lifecycle:
        1. **Warmup** — the pool observes allocation requests and creates new
           buffers on first encounter.  Subsequent requests for the same key
           reuse existing free buffers or grow the bucket.
        2. **Steady state / capture** — all requests are served from
           pre-allocated buffers.  No new CUDA allocations occur.
    """

    # Buffer states
    FREE = 0
    ACQUIRED = 1

    def __init__(self):
        # {(shape, dtype): [ [tensor, state], ... ]}
        self._buckets: dict[tuple, list[list]] = {}
        self._enabled = False
        # data_ptr -> (key, slot_idx) for O(1) release
        self._ptr_to_slot: dict[int, tuple[tuple, int]] = {}

    def enable(self):
        """Enable the pool.  Subsequent acquire/release calls are active."""
        self._enabled = True

    def disable(self):
        """Disable the pool.  acquire() returns None, release() is a no-op."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def acquire(self, shape: tuple, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Acquire a buffer with the given shape and dtype.

        Returns a pre-allocated tensor if one is available, or allocates a new
        one (during warmup).  Returns ``None`` if the pool is disabled.
        """
        if not self._enabled:
            return None

        key = (shape, dtype)
        bucket = self._buckets.get(key)

        if bucket is not None:
            # Try to find a FREE slot
            for slot_idx, slot in enumerate(bucket):
                if slot[1] == self.FREE:
                    slot[1] = self.ACQUIRED
                    return slot[0]
            # All slots busy — allocate a new slot (warmup growth)
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            slot_idx = len(bucket)
            bucket.append([buf, self.ACQUIRED])
            self._ptr_to_slot[buf.data_ptr()] = (key, slot_idx)
            return buf
        else:
            # First time seeing this (shape, dtype) — create bucket
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            self._buckets[key] = [[buf, self.ACQUIRED]]
            self._ptr_to_slot[buf.data_ptr()] = (key, 0)
            return buf

    def release(self, tensor: torch.Tensor):
        """Release a buffer back to the pool.

        Args:
            tensor: A tensor previously returned by :meth:`acquire`.
        """
        if not self._enabled:
            return

        ptr = tensor.data_ptr()
        location = self._ptr_to_slot.get(ptr)
        if location is None:
            # Tensor not from this pool — ignore silently.
            # This happens for tensors allocated before the pool was enabled
            # (e.g. during the very first warmup microbatch).
            return

        key, slot_idx = location
        slot = self._buckets[key][slot_idx]
        assert slot[1] == self.ACQUIRED, (
            f"ActivationPool: double-release detected for "
            f"buffer at {ptr:#x} with key={key}"
        )
        slot[1] = self.FREE

    def owns(self, tensor: torch.Tensor) -> bool:
        """Check if a tensor was allocated by this pool."""
        return tensor.data_ptr() in self._ptr_to_slot

    def stats(self) -> dict:
        """Return pool statistics for debugging."""
        total = 0
        acquired = 0
        memory_bytes = 0
        for key, bucket in self._buckets.items():
            shape, dtype = key
            elem_size = torch.tensor([], dtype=dtype).element_size()
            numel = 1
            for d in shape:
                numel *= d
            for slot in bucket:
                total += 1
                if slot[1] == self.ACQUIRED:
                    acquired += 1
                memory_bytes += numel * elem_size
        return {
            "total_buffers": total,
            "acquired": acquired,
            "free": total - acquired,
            "num_keys": len(self._buckets),
            "memory_mb": memory_bytes / (1024 * 1024),
        }

    def reset(self):
        """Reset all slots to FREE.  Call between training steps if needed."""
        for bucket in self._buckets.values():
            for slot in bucket:
                slot[1] = self.FREE


def set_activation_pool(pool: Optional["ActivationPool"]):
    """Set the global activation pool for combined_1f1b schedules."""
    global _ACTIVATION_POOL
    _ACTIVATION_POOL = pool


def get_activation_pool() -> Optional["ActivationPool"]:
    """Get the global activation pool, or None if not set."""
    return _ACTIVATION_POOL
