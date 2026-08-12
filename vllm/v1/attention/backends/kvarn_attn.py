# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN attention backend.

KV-cache compression by Hadamard rotation + iterative variance-normalization
(Sinkhorn-like) + asymmetric RTN. K is quantized per-channel, V per-token —
KIVI orientation. The variance-normalization tile equals the vLLM
``block_size`` (default and only supported value in this PR: ``128``).

Cache layout (per block, per kv-head, ``head_dim=128, k_bits=4, v_bits=4``):
  17920 B = 8192 (K packed) + 256 + 256 + 256  (K absorbed scales + zp + s_row)
          + 8192 (V packed) + 256 + 256 + 256  (V s_col + absorbed s_row + zp)

vLLM-shape reinterpretation: ``(num_blocks, block_size=128, num_kv_heads, 140)``
where ``140 = 17920 / 128``. The 128-slot middle dim has no semantic per-token
meaning — KVarN treats each ``kv_cache[block, :, head, :].view(-1)`` as one
flat 17920-byte tile record. The slot dim is preserved only to satisfy
vLLM's KV-cache allocator (which expects 4D for non-MLA layouts) and to keep
``slot_mapping`` arithmetic uniform with other backends.

Implementation outline:
  - `do_kv_cache_update` buffers incoming fp16 K/V in a per-block staging dict
    (keyed by block_id). When a block fills to 128 tokens, it rotates by
    Hadamard, calls `kvarn_store_tile_{k,v}` (Stage-3a validated), and writes
    the packed 17920-byte record into the cache.
  - `forward` has three branches: pure-prefill first chunk (raw K/V →
    FlashAttention or vLLM Triton), pure-decode (KVarN Triton decode),
    mixed batch (split decode / prefill).
  - Cached multi-query continuations materialize rotated K/V into shared fp16
    scratch and run a vLLM Triton continuation kernel; small long-context MTP
    verify steps keep the fused KVarN verify kernel.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.kvarn_decode import (
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_v,
)
from vllm.v1.attention.ops.kvarn_store import (
    kvarn_store_tile_k,
    kvarn_store_tile_k_batch_from_sinkhorn,
    kvarn_store_tile_v,
    kvarn_store_tile_v_batch_from_sinkhorn,
)
from vllm.v1.attention.ops.triton_kvarn_decode import kvarn_decode_attention
from vllm.v1.attention.ops.triton_prefill_attention import (
    context_attention_fwd,
    context_attention_fwd_with_kv_lens,
)
from vllm.v1.attention.ops.triton_kvarn_sinkhorn import kvarn_sinkhorn_triton

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


# ──────────────────────────────────────────────────────────────────────────────
# Hadamard cache (one D×D matrix per (head_dim, device))
# ──────────────────────────────────────────────────────────────────────────────


@functools.cache
def _hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    """Sylvester Hadamard, normalised, cached per (d, device)."""
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str)).float()


def _build_hadamard(d: int, device: torch.device) -> torch.Tensor:
    return _hadamard_cached(d, str(torch.device(device)))


def _sinkhorn_pack_kv(K_tiles, V_tiles, cfg):
    """Sinkhorn-balance + pack a batch of K and V tiles into int4 stores.

    K_tiles is [N, D, group] (absorb axis = channel), V_tiles is [N, group, D]
    (absorb axis = token). When D == group (head_dim 128) the two have the same
    [R, C] shape, so we fuse them into ONE Triton Sinkhorn launch. When D != group
    (e.g. head_dim 256, group 128) the tiles are non-square and have different
    [R, C] — kvarn_sinkhorn_triton takes R, C as per-launch constexpr — so K and V
    must be balanced in SEPARATE launches. (A single torch.cat here assumed square
    and broke at head_dim=256.)"""
    if K_tiles.shape[1:] == V_tiles.shape[1:]:
        nk = K_tiles.shape[0]
        bal, sc, sr = kvarn_sinkhorn_triton(
            torch.cat([K_tiles, V_tiles], dim=0), iterations=cfg.sinkhorn_iters,
        )
        K_out = kvarn_store_tile_k_batch_from_sinkhorn(
            bal[:nk], sc[:nk], sr[:nk], bits=cfg.key_bits)
        V_out = kvarn_store_tile_v_batch_from_sinkhorn(
            bal[nk:], sc[nk:], sr[nk:], bits=cfg.value_bits)
    else:
        kbal, ksc, ksr = kvarn_sinkhorn_triton(K_tiles, iterations=cfg.sinkhorn_iters)
        vbal, vsc, vsr = kvarn_sinkhorn_triton(V_tiles, iterations=cfg.sinkhorn_iters)
        K_out = kvarn_store_tile_k_batch_from_sinkhorn(
            kbal, ksc, ksr, bits=cfg.key_bits)
        V_out = kvarn_store_tile_v_batch_from_sinkhorn(
            vbal, vsc, vsr, bits=cfg.value_bits)
    return K_out, V_out


# ──────────────────────────────────────────────────────────────────────────────
# Backend metadata classes
# ──────────────────────────────────────────────────────────────────────────────


class KVarNAttentionBackend(AttentionBackend):
    """Attention backend using KVarN KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "kvarn_k4v4_g128",
        "kvarn_k4v2_g128",
        "kvarn_k4v4_g64",
        "kvarn_k4v2_g64",
    ]

    @staticmethod
    def get_name() -> str:
        return "KVARN"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # One vLLM block == one KVarN tile (cfg.group). Supported tile sizes are
        # the distinct `group` values across the registered presets (64, 128).
        from vllm.model_executor.layers.quantization.kvarn.config import (
            KVARN_PRESETS,
        )
        return sorted({p["group"] for p in KVARN_PRESETS.values()})

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # The active preset pins the tile size (..._g64 / ..._g128), and
        # get_kv_cache_shape asserts block_size == cfg.group. The generic
        # fallback returns the MINIMUM supported size (64) whenever the
        # framework default (16) is unsupported — which breaks any g128 preset
        # run without an explicit --block-size (a g128 deployment then builds
        # its cache with 64-token kernel blocks and dies on the assert, e.g.
        # hybrid models without spec decode). Prefer the preset's group.
        from vllm.config.vllm import get_current_vllm_config
        from vllm.model_executor.layers.quantization.kvarn.config import (
            KVARN_PRESETS,
        )
        try:
            cache_dtype = get_current_vllm_config().cache_config.cache_dtype
        except Exception:
            cache_dtype = None
        if isinstance(cache_dtype, str) and cache_dtype in KVARN_PRESETS:
            return KVARN_PRESETS[cache_dtype]["group"]
        return super().get_preferred_block_size(default_block_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["KVarNAttentionImpl"]:
        return KVarNAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["KVarNMetadataBuilder"]:
        return KVarNMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "kvarn_k4v4_g128",
    ) -> tuple[int, ...]:
        """3D shape: one contiguous ``tile_bytes_aligned`` record per (block, head).

        Unlike TurboQuant's per-token slot, KVarN's scales are tile-shared,
        so one block per head is a single 17920-byte record. The natural
        shape is therefore ``(num_blocks, num_kv_heads, tile_bytes_aligned)``
        — no leading 2 (K and V share the record), and no per-position dim.

        The total bytes per block (= ``num_kv_heads * tile_bytes_aligned``)
        equals ``block_size * num_kv_heads * slot_size`` from
        ``TQFullAttentionSpec.page_size_bytes`` when ``slot_size = tile_bytes
        / block_size``, so vLLM's memory accounting works unchanged.
        """
        from vllm.model_executor.layers.quantization.kvarn.config import (
            KVarNConfig,
        )

        cfg = KVarNConfig.from_cache_dtype(cache_dtype_str, head_size)
        assert block_size == cfg.group, (
            f"KVarN requires block_size ({block_size}) == group ({cfg.group})."
        )
        return (num_blocks, num_kv_heads, cfg.tile_bytes_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("kvarn_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (128, 256, 512)

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Multimodal models (e.g. Gemma-4) set use_mm_prefix; text generation
        # never materializes mm tokens so KVarN decode is unaffected. (Image/audio
        # prefix full-attention correctness is unverified — text-only validated.)
        return True


@dataclass
class KVarNMetadata(AttentionMetadata):
    """Metadata for KVarN attention (mirrors ``TurboQuantMetadata``)."""

    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0
    # True if any multi-query request (query_len > 1) also has cached context
    # (seq_len > query_len): a speculative-decode verify step or a chunked-
    # prefill continuation. Such steps MUST attend over the cached K/V, so they
    # route to the context-aware path rather than _prefill_first_chunk (which
    # assumes a fresh prompt, cached_len == 0). Computed once in build() from
    # CPU arrays (no GPU sync).
    has_cached_multiquery: bool = False
    # Precomputed once per batch in the metadata builder and reused across all
    # 28+ layer forward calls. Saves 28× .tolist() syncs per decode token.
    seq_lens_cpu: list[int] | None = None
    block_table_cpu: list[list[int]] | None = None
    slot_mapping_cpu: list[int] | None = None
    query_start_locs_cpu: list[int] | None = None
    query_lens_cpu: list[int] | None = None
    # Slot mapping used by KVarN's fp16 pool store. During MTP verify, only
    # the real decode token is durable; draft tokens are masked out here and
    # stored in the separate MTP draft scratch below.
    store_slot_mapping: torch.Tensor | None = None
    store_slot_mapping_cpu: list[int] | None = None
    # Per-token scratch index for MTP draft K/V. -1 means "not a draft token".
    # The metadata builder allocates these indices from a per-cache-group
    # scratch allocator, keyed by physical slot_mapping so the next step can
    # promote accepted draft tokens into the durable tail pool after row
    # reordering.
    draft_store_indices: torch.Tensor | None = None
    draft_store_indices_cpu: list[int] | None = None
    # K/V lengths to materialize from persistent cache before appending current
    # raw query K/V. Equals seq_lens except for speculative rows, where it is
    # the committed prefix length (seq_len - query_len).
    fa_build_seq_lens: torch.Tensor | None = None
    has_transient_query_kv: bool = False
    transient_query_kv_rows_cpu: list[bool] | None = None
    # Stage α-2 capture-correct decode metadata. The block_table-driven
    # build-packed-KV kernel reads block_table / seq_lens / fa_cu_seqlens_k
    # directly (all PERSISTENT buffers updated in-place by the builder), so a
    # captured CUDA graph sees fresh data on every replay.
    fa_cu_seqlens_q: torch.Tensor | None = None       # [B+1] int32 (persistent)
    fa_cu_seqlens_k: torch.Tensor | None = None       # [B+1] int32 (persistent prefix sum of seq_lens)
    fa_cu_seqlens_k_cpu: list[int] | None = None
    fa_total_k: int = 0                               # last valid K_packed token offset
    fa_max_blocks_per_req: int = 0                    # ceil(max_model_len / group): grid dim
    fa_max_seqlen_k_fixed: int = 0                    # = max_model_len; fixed FA grid bound
    # Verify (spec-as-decode) plan: one virtual kernel row per decode-portion
    # query token. Persistent buffers (pointers baked into captured graphs),
    # filled CPU-side in build(). None when the decode portion is single-token.
    vq_req: torch.Tensor | None = None                # [num_decode_tokens] int32 block-table row
    vq_seqlen: torch.Tensor | None = None             # [num_decode_tokens] int32 causal length
    vq_qlen: int = 0                                  # uniform decode query len (>=2), else 0


class KVarNMetadataBuilder(AttentionMetadataBuilder[KVarNMetadata]):
    """Builds ``KVarNMetadata`` from scheduler output."""

    # KVarN MTP verify keeps draft K/V in a separate scratch pool and promotes
    # only accepted tokens on the next step. The materialized verify path still
    # appends current query K/V from activations using per-step CPU offsets, so
    # it must run eager. Single-token decode remains graph-capturable.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # spec-as-decode: verify steps (query_len <= 1 + num_spec) classify
        # as decodes and carry a vq plan (see build()) for the fused verify
        # kernel; the threshold is derived from speculative_config by the
        # base helper.
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=False,
        )
        # KV-cache-group key, must match KVarNAttentionImpl._group_key for this
        # group's layers so the builder mutates the right group's slot allocator.
        # (head_size, num_kv_heads, sliding_window) — see impl._group_key.
        # TRUE per-group identity = this builder's exact layer set. A config
        # proxy (head,kv,sw) is NOT enough: vLLM splits same-config layers into
        # multiple groups (Gemma-4's repeating pattern -> 5 sliding groups all
        # head256/16kv/1024), each with its own block_id space. The builder tags
        # its impls with this key in build() (impls don't reliably carry a name).
        self._layer_names = list(layer_names)
        self._layer_names_set = set(self._layer_names)
        self._group_key = tuple(sorted(self._layer_names))
        # Stage α-2: per-block fill tracking — block_id -> tokens present in
        # the pool for that block after the current step. Keyed by PHYSICAL
        # block (never by request or by the sink block id): vLLM's prefix
        # caching shares physical blocks across live requests and recycles ids
        # across finished ones, so any request-identity proxy collides under
        # sharing (the issue #10 repetition-collapse / stale-tile class). A
        # partial block has exactly one writer, so the value has a single
        # source. Drives flush-on-reclaim: a finished request's complete block
        # must be flushed (a future prefix-cache hit may read it), a partial
        # one is safe to discard (vLLM never prefix-caches partial blocks).
        self._block_fill: dict[int, int] = {}
        # Max model length (for the fixed FA grid bound + max_blocks_per_req).
        try:
            self._max_model_len = vllm_config.model_config.max_model_len
        except Exception:
            self._max_model_len = 4096
        try:
            self._max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        except Exception:
            self._max_num_seqs = 256
        try:
            self._max_num_batched_tokens = (
                vllm_config.scheduler_config.max_num_batched_tokens)
        except Exception:
            self._max_num_batched_tokens = 8192

        # KVarN tile / group size (= vLLM block size). Sourced from the configured
        # kv-cache dtype so non-128 groups (e.g. g64) drive the flush + slot math
        # in build() correctly. Every storage / kernel path already reads
        # cfg.group; this is the one place the builder needs it without an impl
        # handle. Falls back to 128 if it cannot be parsed.
        self._group = 128
        try:
            from vllm.model_executor.layers.quantization.kvarn.config import (
                KVarNConfig,
            )
            _cd = vllm_config.cache_config.cache_dtype
            _hd = vllm_config.model_config.get_head_size()
            self._group = KVarNConfig.from_cache_dtype(_cd, _hd).group
        except Exception:
            self._group = 128
        speculative_config = getattr(vllm_config, "speculative_config", None)
        self._has_speculative_config = speculative_config is not None
        self._max_speculative_query_len = 1 + int(getattr(
            speculative_config, "num_speculative_tokens", 0) or 0)
        # Persistent cu_seqlens buffers (allocated lazily in build()).
        self._cu_seqlens_q_buf: torch.Tensor | None = None
        self._cu_seqlens_k_buf: torch.Tensor | None = None
        self._cu_seqlens_q_host: torch.Tensor | None = None
        self._cu_seqlens_k_host: torch.Tensor | None = None
        # Persistent verify-plan buffers (allocated lazily in build()).
        self._vq_req_buf: torch.Tensor | None = None
        self._vq_seqlen_buf: torch.Tensor | None = None
        self._vq_req_host: torch.Tensor | None = None
        self._vq_seqlen_host: torch.Tensor | None = None
        self._store_slot_mapping_buf: torch.Tensor | None = None
        self._store_slot_mapping_host: torch.Tensor | None = None
        self._draft_store_indices_buf: torch.Tensor | None = None
        self._draft_store_indices_host: torch.Tensor | None = None
        self._fa_build_seq_lens_buf: torch.Tensor | None = None
        self._fa_build_seq_lens_host: torch.Tensor | None = None
        self._init_static_buffers(device)

    def _init_static_buffers(self, device: torch.device) -> None:
        """Allocate builder-owned metadata buffers at construction time.

        vLLM's scheduler caps requests and batched tokens, so these sizes are
        fixed for the lifetime of the worker. Growing them from build() would
        create serving-time allocations that are invisible to the initial KVarN
        memory profile.
        """
        seq_cap = max(int(self._max_num_seqs) + 1, 1)
        token_cap = max(int(self._max_num_batched_tokens),
                        int(self._max_num_seqs), 1)
        self._cu_seqlens_q_buf = torch.empty(
            seq_cap, dtype=torch.int32, device=device)
        self._cu_seqlens_k_buf = torch.empty(
            seq_cap, dtype=torch.int32, device=device)
        self._cu_seqlens_q_host = torch.empty(
            seq_cap, dtype=torch.int32, pin_memory=True)
        self._cu_seqlens_k_host = torch.empty(
            seq_cap, dtype=torch.int32, pin_memory=True)
        self._vq_req_buf = torch.empty(
            token_cap, dtype=torch.int32, device=device)
        self._vq_seqlen_buf = torch.empty(
            token_cap, dtype=torch.int32, device=device)
        self._vq_req_host = torch.empty(
            token_cap, dtype=torch.int32, pin_memory=True)
        self._vq_seqlen_host = torch.empty(
            token_cap, dtype=torch.int32, pin_memory=True)
        self._store_slot_mapping_buf = torch.empty(
            token_cap, dtype=torch.long, device=device)
        self._store_slot_mapping_host = torch.empty(
            token_cap, dtype=torch.long, pin_memory=True)
        self._draft_store_indices_buf = torch.empty(
            token_cap, dtype=torch.int32, device=device)
        self._draft_store_indices_host = torch.empty(
            token_cap, dtype=torch.int32, pin_memory=True)
        self._fa_build_seq_lens_buf = torch.empty(
            seq_cap - 1, dtype=torch.int32, device=device)
        self._fa_build_seq_lens_host = torch.empty(
            seq_cap - 1, dtype=torch.int32, pin_memory=True)

    def _is_spec_decode_row(
        self,
        b: int,
        seq_len: int,
        query_len: int,
        is_prefilling_cpu: list[bool] | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> bool:
        """True for MTP verify rows whose draft K/V must stay transient."""
        if query_len <= 1 or seq_len <= query_len:
            return False
        if (is_prefilling_cpu is not None
                and b < len(is_prefilling_cpu)
                and is_prefilling_cpu[b]):
            return False
        if num_decode_draft_tokens_cpu is not None and b < len(num_decode_draft_tokens_cpu):
            drafts = int(num_decode_draft_tokens_cpu[b])
            return drafts > 0 and query_len == drafts + 1
        return (self._has_speculative_config
                and query_len <= self._max_speculative_query_len)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> KVarNMetadata:
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    ):
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )
        # Pre-materialise CPU views ONCE per batch. Every layer's forward()
        # would otherwise re-issue these syncs (28+ syncs/token for Qwen3-0.6B).
        # Use the framework's cached CPU copy of seq_lens (cam.seq_lens_cpu) to
        # avoid an extra GPU->CPU sync per step (issue #15 build-overhead).
        _slc = getattr(cam, "seq_lens_cpu", None)
        seq_lens_cpu = (_slc.tolist() if _slc is not None else cam.seq_lens.tolist())
        # Per-request query length this step (already on CPU; no extra sync).
        # Used by flush detection to compute committed tokens so speculative
        # tokens that may still be rejected are never quantized permanently.
        _qsl = getattr(cam, "query_start_loc_cpu", None)
        query_start_locs_cpu = None
        if _qsl is not None:
            query_start_locs_cpu = _qsl.tolist()
            query_lens_cpu = [
                query_start_locs_cpu[i + 1] - query_start_locs_cpu[i]
                for i in range(len(query_start_locs_cpu) - 1)
            ]
        else:
            query_lens_cpu = [1] * len(seq_lens_cpu)
        is_prefilling_cpu = None
        if cam.is_prefilling is not None:
            is_prefilling_cpu = [
                bool(x) for x in cam.is_prefilling.tolist()
            ]
        # block_table as a numpy 2-D array (C-backed, lazy element access) rather
        # than .tolist(): the full B×max_blocks nested-list build was ~7 ms/step
        # at B=256 and dominated build() once the flush was vectorized (issue #15).
        # We only touch column 0 (sinks) + a few per-request entries, so numpy's
        # O(1) indexing avoids materializing ~8k Python ints every step.
        block_table_np = cam.block_table_tensor.cpu().numpy()
        slot_mapping_cpu = cam.slot_mapping.tolist()
        bt_rows = block_table_np.shape[0]
        bt_cols = block_table_np.shape[1] if block_table_np.ndim == 2 else 0
        device = cam.seq_lens.device

        GROUP = self._group  # KVarN tile size (= block size); 64 or 128
        store_slot_mapping_cpu = list(slot_mapping_cpu)
        draft_store_indices_cpu = [-1] * len(slot_mapping_cpu)
        accepted_tokens_cpu = [1] * len(seq_lens_cpu)
        if num_accepted_tokens is not None:
            accepted_tokens_cpu = [
                int(x) for x in num_accepted_tokens[:len(seq_lens_cpu)]
                .detach().cpu().tolist()
            ]
        fa_build_seq_lens_cpu: list[int] = []
        transient_query_kv_rows_cpu: list[bool] = []
        current_draft_slot_mappings: list[tuple[int, int]] = []
        promote_slot_mappings: list[int] = []
        for b, sl in enumerate(seq_lens_cpu):
            q_len = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
            is_spec_row = self._is_spec_decode_row(
                b, sl, q_len, is_prefilling_cpu, num_decode_draft_tokens_cpu)
            transient_query_kv_rows_cpu.append(is_spec_row)
            committed = max(sl - q_len, 0)
            fa_build_seq_lens_cpu.append(committed if is_spec_row else sl)
            q_start = (
                query_start_locs_cpu[b]
                if query_start_locs_cpu is not None and b < len(query_start_locs_cpu)
                else b
            )
            if is_spec_row:
                # Only the first token in an MTP verify row is unconditionally
                # committed. Draft tokens are written to the per-group scratch
                # pool and promoted on the next step once vLLM reports how many
                # draft tokens were accepted.
                for j in range(1, q_len):
                    idx = q_start + j
                    if idx < len(store_slot_mapping_cpu):
                        sm = store_slot_mapping_cpu[idx]
                        store_slot_mapping_cpu[idx] = -1
                        if sm >= 0:
                            current_draft_slot_mappings.append((idx, sm))

            accepted = (
                accepted_tokens_cpu[b] if b < len(accepted_tokens_cpu) else 1)
            if accepted > 1 and b < bt_rows and bt_cols > 0:
                # accepted includes the real token at offset 0. The accepted
                # draft positions are the tail of the already-computed prefix
                # immediately before this step's query tokens.
                first_pos = max(committed - accepted + 1, 0)
                last_pos = max(committed - 1, -1)
                row = block_table_np[b]
                for pos in range(first_pos, last_pos + 1):
                    k = pos // GROUP
                    if k >= bt_cols:
                        break
                    bid = int(row[k])
                    if bid >= 0:
                        promote_slot_mappings.append(bid * GROUP + (pos % GROUP))
        has_transient_query_kv = any(transient_query_kv_rows_cpu)

        # ── Stage α-2: capture-correct metadata ──────────────────────────
        # The decode driver uses ONE block_table-driven kernel that reads the
        # PERSISTENT block_table / seq_lens / cu_seqlens directly, so no
        # per-step derived task tensors (which would be stale under graph
        # replay). We only need cu_seqlens_k (prefix sum of seq_lens) and
        # cu_seqlens_q (= arange(B+1)), both kept in PERSISTENT buffers and
        # updated in place so captured graphs see fresh values.
        B = len(seq_lens_cpu)
        cu_seqlens_k_h = [0]
        for sl in seq_lens_cpu:
            cu_seqlens_k_h.append(cu_seqlens_k_h[-1] + sl)

        if (self._fa_build_seq_lens_buf is None
                or self._fa_build_seq_lens_buf.shape[0] < B):
            raise RuntimeError(
                "KVarN static build-lens buffer too small: "
                f"need {B}, have "
                f"{0 if self._fa_build_seq_lens_buf is None else self._fa_build_seq_lens_buf.shape[0]}. "
                "Increase scheduler max_num_seqs before startup.")
        for i, sl in enumerate(fa_build_seq_lens_cpu):
            self._fa_build_seq_lens_host[i] = sl
        fa_build_seq_lens = self._fa_build_seq_lens_buf[:B]
        fa_build_seq_lens.copy_(
            self._fa_build_seq_lens_host[:B], non_blocking=True)

        # ── Stage α-2: assign pool slots for every block_id touched this
        # step. The allocator state is class-level on KVarNAttentionImpl
        # and we mutate it here (in the builder, outside any captured
        # region). do_kv_cache_update then only READS block_to_slot_t.
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionImpl  # local import
        # Pool slots are needed ONLY for blocks that physically live in the fp16
        # tail pool: each request's sink (block_table[r][0], kept fp16 for the
        # request's lifetime) and the blocks receiving writes THIS step —
        # tokens committed..seq_len-1 land in do_kv_cache_update after the
        # builder. Flushed history blocks live in the int4 cache, carry
        # pool_slot=-1, and are dequantized in-kernel.
        #
        # Sharing-safe lifecycle (prefix caching + chunked prefill + spec
        # decode): everything below derives from per-step facts —
        #   committed = seq_len - query_len   (tokens written BEFORE this step)
        #   dict_map membership = "block is unflushed" (ground truth: a flush
        #     frees the slot, so a slot-holding block below the committed
        #     boundary is exactly a full-but-unflushed block)
        #   _block_fill[bid] = tokens the pool holds for bid after this step
        # A cache-hit request's context blocks receive no writes, so they
        # correctly need no slots (their tiles are already int4). Anything in
        # the allocator NOT needed this step belongs to a finished request and
        # is reclaimed below: complete blocks are FLUSHED (a future prefix-
        # cache hit must find a valid int4 tile), partial ones discarded.
        blocks_needed: set[int] = set()
        for b in range(B):
            if b >= bt_rows:
                break
            sl = seq_lens_cpu[b]
            if bt_cols == 0 or sl <= 0:
                continue
            row = block_table_np[b]
            q_len = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
            committed = max(sl - q_len, 0)
            safe_query_len = 1 if transient_query_kv_rows_cpu[b] else q_len
            safe_seq_len = committed + safe_query_len
            if safe_seq_len <= committed:
                continue
            # Blocks written this step. Record how full each will be AFTER the
            # step: if its owner finishes on the step that fills it, the
            # reclaim below must flush it (not discard).
            for k in range(committed // GROUP,
                           min((safe_seq_len - 1) // GROUP, bt_cols - 1) + 1):
                bid = int(row[k])
                if bid >= 0:
                    blocks_needed.add(bid)
                    self._block_fill[bid] = (
                        min(safe_seq_len, (k + 1) * GROUP) - k * GROUP)
        for s in store_slot_mapping_cpu:
            if s >= 0:
                blocks_needed.add(s // GROUP)
        for s in promote_slot_mappings:
            if s >= 0:
                blocks_needed.add(s // GROUP)

        gk = self._group_key
        # Claim THIS group's impls by layer name (set on the impl in
        # Attention.__init__) and tag them with the true group key, so their
        # _ensure_pool / store paths use this group's slot allocator + mirror.
        group_impls = [i for i in KVarNAttentionImpl._all_impls
                       if getattr(i, "layer_name", None) in self._layer_names_set]
        for i in group_impls:
            i._group_key = gk
        if group_impls:
            # Ensure pool + lookup tensors exist for this device.
            num_blocks_hint = max(blocks_needed, default=0) + 1
            for impl in group_impls:
                impl._ensure_pool(device, num_blocks_hint=num_blocks_hint)
            impl0 = group_impls[0]
            mkey = (device, gk)
            b2s_t = KVarNAttentionImpl._block_to_slot_t_per_device[mkey]
            is_sink_t = KVarNAttentionImpl._is_sink_t_per_device[mkey]
            dict_map = KVarNAttentionImpl._block_to_slot_dict[gk]
            free_slots = KVarNAttentionImpl._free_slots[gk]
            sinks = KVarNAttentionImpl._global_sink_blocks[gk]
            pending_cow_dst = KVarNAttentionImpl._pending_cow_dst_blocks.setdefault(
                gk, set())

            # ORDER MATTERS: mark sinks → FLUSH (frees just-completed blocks'
            # slots) → ALLOCATE (the new tails, reusing the freed slots). Doing
            # the flush before allocation caps the live-slot peak at 2·B
            # (one sink + one in-progress tail per request). Allocating first
            # would transiently need 3·B when every request crosses a block
            # boundary in lockstep (sink + pending-flush full block + new tail)
            # → "pool exhausted" at large batch.

            # (1) Mark per-request sink blocks (block_table[r][0]). A block is
            # an fp16 sink only while its data lives in the pool: a fresh
            # prefill writes block 0 this step (it is in blocks_needed) and
            # keeps it fp16 for the request's lifetime; an existing sink keeps
            # its slot via blocks_needed. A prefix-cache-hit request whose
            # block 0 was already reclaimed (flushed to int4) must NOT re-mark
            # it: its data lives in the int4 tile (slot -1) and every kernel
            # reads it there like any history block. Re-marking would allocate
            # an EMPTY pool slot that is never written (cache hits skip those
            # tokens) and attention would read garbage for the whole first
            # block — the issue #10 repetition-loops on multi-turn chat.
            row0_set: set[int] = set()
            for b in range(B):
                if b >= bt_rows or bt_cols == 0:
                    break
                s0 = int(block_table_np[b, 0])
                if s0 < 0:
                    continue
                row0_set.add(s0)
                if s0 in sinks:
                    blocks_needed.add(s0)          # live sink keeps its slot
                elif s0 in blocks_needed:          # written this step → fresh sink
                    sinks.add(s0)
                    if s0 < is_sink_t.shape[0]:
                        is_sink_t[s0] = True

            # (2) Flush detection (Stage α-2 Step B).
            # CRITICAL timing: token (k+1)*GROUP-1 (the one that completes
            # block k) is written during THIS step's do_kv_cache_update, which
            # runs AFTER the builder. So at builder time the pool only holds
            # tokens already committed before this step. That committed count is
            # `seq_len - query_len` (this step's query tokens are written later),
            # i.e. exactly num_computed_tokens. We flush against THAT, never the
            # full `sl`.
            #
            # Why not the full `sl` (or the previous step's `sl`): under
            # speculative decoding (MTP / draft) a step appends `num_spec+1`
            # tokens at once and seq_len jumps by a VARIABLE accepted amount.
            # Current draft K/V lives in MTP scratch, not the tail pool; accepted
            # drafts are promoted at the start of a later step. Quantizing a
            # block to int4 is PERMANENT, so flushing is still based on the
            # committed boundary, after promotion, and never on optimistic
            # speculative length.
            #
            # Walk each row BACKWARD from the committed boundary while blocks
            # still hold pool slots — those are exactly the full-but-unflushed
            # blocks. The walk stops at the first slotless block (flushes
            # happen in order, so everything earlier is already int4) and never
            # touches k=0 (a live request's sink stays fp16; finished requests'
            # sinks are handled by the reclaim below). Idempotent under prefix
            # sharing: a co-owner finds the block already queued (or slotless)
            # and stops — no per-request state to collide.
            flush_block_ids: list[int] = []
            flush_seen: set[int] = set()
            for b in range(B):
                if b >= bt_rows or bt_cols == 0:
                    break
                sl = seq_lens_cpu[b]
                row = block_table_np[b]
                if sl <= 0:
                    continue
                q_len = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
                committed_len = max(sl - q_len, 0)    # tokens already in pool & accepted
                k = min(committed_len // GROUP - 1, bt_cols - 1)
                while 1 <= k:
                    bid = int(row[k])
                    if (bid < 0 or bid in flush_seen or bid in sinks
                            or bid not in dict_map):
                        break
                    flush_seen.add(bid)
                    flush_block_ids.append(bid)
                    k -= 1

            # (2b) Reclaim slot-holding blocks neither written this step nor
            # queued above: they belong to finished (or preempted) requests.
            # Every COMPLETE block is flushed, including sink/block-0. Keeping
            # a finished sink fp16-resident is unsafe without a vLLM block
            # lifetime generation: the physical block id can later be recycled,
            # while KVarN would still map it to the old fp16 pool slot. Prefix-
            # cache hits then read stale fp16 data instead of the copied int4
            # backing cache. A PARTIAL block is discarded because vLLM never
            # prefix-caches partial blocks.
            discard_ids: list[int] = []
            for bid in [b for b in dict_map
                        if b not in blocks_needed and b not in flush_seen]:
                full = self._block_fill.get(bid, 0) >= GROUP
                if full:
                    flush_seen.add(bid)
                    flush_block_ids.append(bid)
                else:
                    discard_ids.append(bid)
                if bid in sinks:                   # finished request's partial sink
                    sinks.discard(bid)
                    if bid < is_sink_t.shape[0]:
                        is_sink_t[bid] = False

            # Trigger the flush on every layer's pool. Each impl quantises its
            # own pool[slot] into its own kv_cache (ref cached on first
            # forward), then frees the slot below. Runs eagerly here, before
            # the captured forward replay.
            if flush_block_ids:
                # One batched Sinkhorn + RTN over ALL (layer, block) flush tiles
                # — replaces 48×N_blocks individual launches. Numerically
                # identical (per-tile-independent ops) → no accuracy change.
                flush_pairs = []
                for impl in group_impls:
                    kvc = getattr(impl, "_kv_cache_ref", None)
                    if kvc is None:
                        continue
                    for bid in flush_block_ids:
                        flush_pairs.append((impl, bid, kvc))
                KVarNAttentionImpl._batched_flush(flush_pairs)
                # Free the flushed blocks' slots so the allocation below can
                # reuse them (they now live in int4; pool_slot → -1).
                for bid in flush_block_ids:
                    slot = dict_map.pop(bid, None)
                    self._block_fill.pop(bid, None)
                    if slot is not None:
                        free_slots.append(slot)
                        if bid < b2s_t.shape[0]:
                            b2s_t[bid] = -1

            # Free the discarded (partial, never-cacheable) blocks' slots.
            for bid in discard_ids:
                slot = dict_map.pop(bid)
                self._block_fill.pop(bid, None)
                free_slots.append(slot)
                if bid < b2s_t.shape[0]:
                    b2s_t[bid] = -1

            # (3) Allocate slots for any new block_ids (sinks + new tails).
            cow_init_pairs: list[tuple[int, int]] = []
            for bid in blocks_needed:
                if bid not in dict_map:
                    if not free_slots:
                        raise RuntimeError(
                            f"KVarN pool exhausted "
                            f"({KVarNAttentionImpl._allocator_pool_size.get(gk)} slots)"
                        )
                    slot = free_slots.pop()
                    dict_map[bid] = slot
                    if bid < b2s_t.shape[0]:
                        b2s_t[bid] = slot
                    KVarNAttentionImpl._max_known_block_id[gk] = max(
                        KVarNAttentionImpl._max_known_block_id.get(gk, 0), bid
                    )
                    if bid in pending_cow_dst:
                        cow_init_pairs.append((bid, slot))
                elif bid in pending_cow_dst:
                    cow_init_pairs.append((bid, dict_map[bid]))

            if cow_init_pairs:
                KVarNAttentionImpl._init_pool_slots_from_cache(
                    group_impls, cow_init_pairs)
                for bid, _ in cow_init_pairs:
                    pending_cow_dst.discard(bid)

            # (4) Promote accepted MTP draft tokens from last step's scratch
            # into the durable fp16 tail pool. `num_accepted_tokens` belongs to
            # the previous speculative step; using the current block table and
            # committed length above gives the physical slots of the accepted
            # draft suffix even if vLLM reordered rows between steps.
            draft_map = KVarNAttentionImpl._draft_slot_to_index[gk]
            draft_free = KVarNAttentionImpl._draft_free_indices[gk]
            for sm in promote_slot_mappings:
                scratch_idx = draft_map.get(sm)
                if scratch_idx is None:
                    continue
                bid = sm // GROUP
                pos = sm % GROUP
                pool_slot = dict_map.get(bid)
                if pool_slot is None:
                    continue
                for impl in group_impls:
                    if (impl._draft_K_scratch is None
                            or impl._draft_V_scratch is None
                            or impl._tail_K_pool is None
                            or impl._tail_V_pool is None):
                        continue
                    impl._tail_K_pool[pool_slot, pos].copy_(
                        impl._draft_K_scratch[scratch_idx])
                    impl._tail_V_pool[pool_slot, pos].copy_(
                        impl._draft_V_scratch[scratch_idx])
                self._block_fill[bid] = max(self._block_fill.get(bid, 0), pos + 1)

            # Free every previous-step scratch row now. Rows not promoted are
            # rejected draft tokens or state for requests that left the batch.
            for _, idx in list(draft_map.items()):
                draft_free.append(idx)
            draft_map.clear()

            # (5) Allocate scratch rows for this step's current draft tokens.
            for token_idx, sm in current_draft_slot_mappings:
                if sm in draft_map:
                    scratch_idx = draft_map[sm]
                else:
                    if not draft_free:
                        raise RuntimeError(
                            "KVarN MTP draft scratch exhausted "
                            f"({KVarNAttentionImpl._draft_scratch_size.get(gk)} slots). "
                            "Increase KVARN_DRAFT_SCRATCH_SLOTS.")
                    scratch_idx = draft_free.pop()
                    draft_map[sm] = scratch_idx
                if token_idx < len(draft_store_indices_cpu):
                    draft_store_indices_cpu[token_idx] = scratch_idx

        if (self._store_slot_mapping_buf is None
                or self._store_slot_mapping_buf.shape[0] < len(store_slot_mapping_cpu)):
            raise RuntimeError(
                "KVarN static store slot-mapping buffer too small: "
                f"need {len(store_slot_mapping_cpu)}, have "
                f"{0 if self._store_slot_mapping_buf is None else self._store_slot_mapping_buf.shape[0]}. "
                "Increase scheduler max_num_batched_tokens before startup.")
        if (self._draft_store_indices_buf is None
                or self._draft_store_indices_buf.shape[0] < len(draft_store_indices_cpu)):
            raise RuntimeError(
                "KVarN static draft-store index buffer too small: "
                f"need {len(draft_store_indices_cpu)}, have "
                f"{0 if self._draft_store_indices_buf is None else self._draft_store_indices_buf.shape[0]}. "
                "Increase scheduler max_num_batched_tokens before startup.")
        for i, slot in enumerate(store_slot_mapping_cpu):
            self._store_slot_mapping_host[i] = slot
        for i, scratch_idx in enumerate(draft_store_indices_cpu):
            self._draft_store_indices_host[i] = scratch_idx
        store_slot_mapping = self._store_slot_mapping_buf[:len(store_slot_mapping_cpu)]
        draft_store_indices = self._draft_store_indices_buf[:len(draft_store_indices_cpu)]
        store_slot_mapping.copy_(
            self._store_slot_mapping_host[:len(store_slot_mapping_cpu)],
            non_blocking=True)
        draft_store_indices.copy_(
            self._draft_store_indices_host[:len(draft_store_indices_cpu)],
            non_blocking=True)

        # ── Persistent cu_seqlens buffers (in-place updated) ─────────────
        # A captured graph bakes in tensor addresses, so cu_seqlens MUST live
        # in fixed buffers updated in place — not recreated each step.
        cap = B + 1
        if self._cu_seqlens_q_buf is None or self._cu_seqlens_q_buf.shape[0] < cap:
            raise RuntimeError(
                "KVarN static cu_seqlens buffer too small: "
                f"need {cap}, have "
                f"{0 if self._cu_seqlens_q_buf is None else self._cu_seqlens_q_buf.shape[0]}. "
                "Increase scheduler max_num_seqs before startup.")
        for i in range(B + 1):
            self._cu_seqlens_q_host[i] = i
            self._cu_seqlens_k_host[i] = cu_seqlens_k_h[i]
        fa_cu_seqlens_q = self._cu_seqlens_q_buf[:B + 1]
        fa_cu_seqlens_k = self._cu_seqlens_k_buf[:B + 1]
        fa_cu_seqlens_q.copy_(self._cu_seqlens_q_host[:B + 1], non_blocking=True)
        fa_cu_seqlens_k.copy_(self._cu_seqlens_k_host[:B + 1], non_blocking=True)

        # ── Verify (spec-as-decode) plan ─────────────────────────────────
        # When the decode portion carries multi-token queries (an MTP verify
        # step; query_len <= reorder threshold), build one virtual kernel row
        # per decode token: its block-table row and its bottom-right causal
        # length (committed + idx + 1). Persistent buffers, CPU-filled here,
        # so the fused verify kernel is CUDA-graph-capturable (pointers stay
        # stable across replays; only values change).
        vq_req_t = vq_seqlen_t = None
        vq_qlen = 0
        if num_decodes > 0 and num_decode_tokens > num_decodes:
            if (self._vq_req_buf is None
                    or self._vq_req_buf.shape[0] < num_decode_tokens):
                raise RuntimeError(
                    "KVarN static verify-plan buffer too small: "
                    f"need {num_decode_tokens}, have "
                    f"{0 if self._vq_req_buf is None else self._vq_req_buf.shape[0]}. "
                    "Increase scheduler max_num_batched_tokens before startup.")
            i = 0
            uniform = query_lens_cpu[0] if num_decodes else 0
            for b in range(num_decodes):
                ql = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
                if ql != uniform:
                    uniform = 0
                committed = max(seq_lens_cpu[b] - ql, 0)
                for j in range(ql):
                    self._vq_req_host[i] = b
                    self._vq_seqlen_host[i] = committed + j + 1
                    i += 1
            # Uniform query length -> the shared-dequant verify kernel (the
            # request's tokens share each block's dequant); this is always the
            # case under uniform-batch graph capture.
            vq_qlen = uniform if uniform >= 2 else 0
            vq_req_t = self._vq_req_buf[:num_decode_tokens]
            vq_seqlen_t = self._vq_seqlen_buf[:num_decode_tokens]
            vq_req_t.copy_(self._vq_req_host[:num_decode_tokens],
                           non_blocking=True)
            vq_seqlen_t.copy_(self._vq_seqlen_host[:num_decode_tokens],
                              non_blocking=True)

        max_blocks_per_req = (self._max_model_len + GROUP - 1) // GROUP

        # A multi-query request with cached context (seq_len > query_len) is a
        # speculative-decode verify step or a chunked-prefill continuation —
        # its query tokens must attend over the cached K/V, not just each other.
        # Detected here from CPU arrays (no GPU sync) so forward() can route it
        # to the context-aware path. Fresh first-chunk prefill has
        # seq_len == query_len on every row → flag stays False.
        has_cached_multiquery = any(
            query_lens_cpu[b] > 1 and seq_lens_cpu[b] > query_lens_cpu[b]
            for b in range(min(B, len(query_lens_cpu)))
        )

        return KVarNMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            has_cached_multiquery=has_cached_multiquery,
            seq_lens_cpu=seq_lens_cpu,
            block_table_cpu=None,  # not consumed downstream; build() uses block_table_np
            slot_mapping_cpu=slot_mapping_cpu,
            query_start_locs_cpu=query_start_locs_cpu,
            query_lens_cpu=query_lens_cpu,
            store_slot_mapping=store_slot_mapping,
            store_slot_mapping_cpu=store_slot_mapping_cpu,
            draft_store_indices=draft_store_indices,
            draft_store_indices_cpu=draft_store_indices_cpu,
            fa_cu_seqlens_q=fa_cu_seqlens_q,
            fa_cu_seqlens_k=fa_cu_seqlens_k,
            fa_cu_seqlens_k_cpu=cu_seqlens_k_h,
            fa_build_seq_lens=fa_build_seq_lens,
            fa_total_k=int(cu_seqlens_k_h[B]) if B < len(cu_seqlens_k_h) else 0,
            fa_max_blocks_per_req=max_blocks_per_req,
            fa_max_seqlen_k_fixed=self._max_model_len,
            vq_req=vq_req_t,
            vq_seqlen=vq_seqlen_t,
            vq_qlen=vq_qlen,
            has_transient_query_kv=has_transient_query_kv,
            transient_query_kv_rows_cpu=transient_query_kv_rows_cpu,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-block fp16 tail buffer (in-progress tile staging)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _BlockTail:
    """In-progress fp16 K/V for one cache block.

    Reset whenever a token with ``position_in_block == 0`` arrives for this
    block_id (handles vLLM's block recycling on request preemption). Evicted
    immediately after a 128-token flush.
    """

    K: torch.Tensor  # [group, num_kv_heads, head_dim] fp16
    V: torch.Tensor  # [group, num_kv_heads, head_dim] fp16
    filled_mask: torch.Tensor = field(repr=False)  # [group] bool — which slots written
    filled_count: int = 0                          # CPU-side counter (avoid .all() sync)


# ──────────────────────────────────────────────────────────────────────────────
# Attention impl
# ──────────────────────────────────────────────────────────────────────────────


class KVarNAttentionImpl(AttentionImpl["KVarNMetadata"]):
    """KVarN attention implementation.

    Slow PyTorch decode for Stage 3b.2 — replaced by Triton in Stage 4.
    """

    supports_quant_query_input: bool = False

    # Shared decode scratch — these are per-step throwaway buffers used by
    # `kvarn_decode_attention`. Sharing across all impl instances (one set
    # per device) avoids 28× memory waste on the per-layer attention.
    # Lazily allocated by `_ensure_pool` on the first non-capture call.
    _shared_q_fp32_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_q_rot_fp32_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_q_rot_fp16_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_out_rot_fp32_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_output_fp32_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_fused_out_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_mid_o_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}    # split-K partials
    _shared_mid_lse_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_fa_K_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_fa_V_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_prefill_out_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_materialized_q_rot_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_materialized_out_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}

    # ── Stage α-2: class-level shared sparse slot allocator ──────────────────
    # Single source of truth across all 28 KVarNAttentionImpl instances:
    #   _block_to_slot_dict[block_id] → slot      (Python, CPU)
    #   _block_to_slot_t_per_device[device][block_id] → slot int32    (GPU mirror)
    #   _is_sink_t_per_device[device][block_id] → bool                (GPU mirror)
    # All allocator mutations happen in KVarNMetadataBuilder.build(), which
    # runs once per step OUTSIDE any captured CUDA-graph region. The captured
    # do_kv_cache_update kernel just reads block_to_slot_t.
    # Allocator state, scoped PER KV-CACHE-GROUP (key = group_key tuple), because
    # block_ids are only unique WITHIN a group. CPU dicts keyed by group_key; GPU
    # mirrors keyed by (device, group_key). See `self._group_key`.
    _block_to_slot_dict: ClassVar[dict[tuple, dict[int, int]]] = {}
    _global_sink_blocks: ClassVar[dict[tuple, set[int]]] = {}
    _free_slots: ClassVar[dict[tuple, list[int]]] = {}
    _allocator_pool_size: ClassVar[dict[tuple, int]] = {}
    _block_to_slot_t_per_device: ClassVar[dict[tuple, torch.Tensor]] = {}
    _is_sink_t_per_device: ClassVar[dict[tuple, torch.Tensor]] = {}
    _max_known_block_id: ClassVar[dict[tuple, int]] = {}
    # MTP draft scratch allocator, scoped per KV-cache group. Keys are physical
    # token slot_mappings (block_id * group + pos). Values are compact scratch
    # indices shared by every layer in the group; each layer owns its actual
    # K/V scratch tensors and the builder promotes accepted slots by index.
    _draft_slot_to_index: ClassVar[dict[tuple, dict[int, int]]] = {}
    _draft_free_indices: ClassVar[dict[tuple, list[int]]] = {}
    _draft_scratch_size: ClassVar[dict[tuple, int]] = {}
    _pending_cow_dst_blocks: ClassVar[dict[tuple, set[int]]] = {}
    # Keys (device, D, group, k_bits, v_bits) whose flush kernels (Sinkhorn +
    # int4 store) have already been JIT-compiled via the pool-init warmup.
    _kernel_warmed: ClassVar[set] = set()

    # Registry of impls so the builder can enumerate per-layer pools when
    # it needs to update sink markers / trigger flushes.
    _all_impls: ClassVar[list["KVarNAttentionImpl"]] = []

    @classmethod
    def _impls_for_group(cls, group_key: tuple) -> list["KVarNAttentionImpl"]:
        """Impls belonging to one KV-cache group (same group_key)."""
        return [i for i in cls._all_impls if i._group_key == group_key]

    @classmethod
    def prepare_kv_cache_block_copies(cls, block_copies) -> None:
        """Make vLLM CoW block copies coherent with KVarN's fp16 pool.

        The generic copy path only copies the compressed KV-cache storage. KVarN
        may still hold a source block's freshest bytes in the fp16 tail pool, so
        flush pool-resident sources first. Destinations are marked for pool
        initialization when the metadata builder allocates their write slot.
        After the forced source flush, retire the source from the fp16 lookup:
        the compressed cache is now authoritative and a stale block->slot mapping
        could later alias a reused pool slot.
        """
        if not block_copies or not cls._all_impls:
            return
        src_ids = {int(c.src_block_id) for c in block_copies}
        dst_ids = {int(c.dst_block_id) for c in block_copies}
        for gk, dict_map in list(cls._block_to_slot_dict.items()):
            cls._pending_cow_dst_blocks.setdefault(gk, set()).update(dst_ids)
            src_to_flush = [bid for bid in src_ids if bid in dict_map]
            if not src_to_flush:
                continue
            flush_pairs = []
            for impl in cls._impls_for_group(gk):
                kvc = getattr(impl, "_kv_cache_ref", None)
                if kvc is None:
                    continue
                for bid in src_to_flush:
                    flush_pairs.append((impl, bid, kvc))
            if not flush_pairs:
                continue
            cls._batched_flush(flush_pairs)
            free_slots = cls._free_slots.get(gk)
            sinks = cls._global_sink_blocks.get(gk)
            for bid in src_to_flush:
                slot = dict_map.pop(bid, None)
                if slot is not None and free_slots is not None:
                    free_slots.append(slot)
                if sinks is not None:
                    sinks.discard(bid)
                for (device, key), b2s_t in list(cls._block_to_slot_t_per_device.items()):
                    if key == gk and bid < b2s_t.shape[0]:
                        b2s_t[bid] = -1
                for (device, key), is_sink_t in list(cls._is_sink_t_per_device.items()):
                    if key == gk and bid < is_sink_t.shape[0]:
                        is_sink_t[bid] = False

    @classmethod
    def _init_pool_slots_from_cache(
        cls,
        group_impls: list["KVarNAttentionImpl"],
        init_pairs: list[tuple[int, int]],
    ) -> None:
        if not group_impls or not init_pairs:
            return
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_dequant_cache_blocks_to_pool_kernel,
        )

        for impl in group_impls:
            kvc = getattr(impl, "_kv_cache_ref", None)
            if (kvc is None or impl._tail_K_pool is None
                    or impl._tail_V_pool is None):
                continue
            device = impl._tail_K_pool.device
            bids = torch.as_tensor([p[0] for p in init_pairs],
                                   dtype=torch.long, device=device)
            slots = torch.as_tensor([p[1] for p in init_pairs],
                                    dtype=torch.long, device=device)
            cfg = impl.kvarn_config
            _kvarn_dequant_cache_blocks_to_pool_kernel[
                (len(init_pairs), impl.num_kv_heads)
            ](
                kvc, bids, slots, impl._tail_K_pool, impl._tail_V_pool,
                kvc.stride(0), kvc.stride(1),
                impl._tail_K_pool.stride(0),
                impl._tail_K_pool.stride(1),
                impl._tail_K_pool.stride(2),
                D=cfg.head_dim, GROUP=cfg.group,
                K_BITS=cfg.key_bits, V_BITS=cfg.value_bits,
                K_PACKED_OFFSET=cfg.k_packed_offset,
                K_S_COL_OFFSET=cfg.k_s_col_offset,
                K_ZP_OFFSET=cfg.k_zp_offset,
                K_S_ROW_OFFSET=cfg.k_s_row_offset,
                V_PACKED_OFFSET=cfg.v_packed_offset,
                V_S_COL_OFFSET=cfg.v_s_col_offset,
                V_S_ROW_OFFSET=cfg.v_s_row_offset,
                V_ZP_OFFSET=cfg.v_zp_offset,
                num_warps=4, num_stages=2,
            )

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        # Sliding-window layers (e.g. Gemma-4: 50/60 layers, window 1024) only
        # attend to the last `sliding_window` keys. Stored so the decode kernel
        # can bound its block loop to the window — without this it reads the FULL
        # history every step (16x too much work + wrong output past the window).
        self.sliding_window = sliding_window or 0
        # KV-cache-group key. KVarN's slot allocator + GPU mirrors are keyed by
        # block_id, but vLLM gives each KV-cache group an INDEPENDENT block_id
        # space. Heterogeneous models put KVarN layers in >1 group (e.g. Gemma-4:
        # sliding head256/16kv + global head512/4kv), so a single global allocator
        # aliases the two groups' block_ids -> wrong slots -> garbage. Scope all
        # allocator state by this key so each group has its own slot space.
        # (head_size, num_kv_heads, sliding_window) uniquely identifies the group
        # and is computable identically by the per-group builder and each impl.
        self._group_key = (head_size, self.num_kv_heads, self.sliding_window)
        if os.environ.get("KVARN_DBG_LAYERS") == "1":
            print(f"[KVARN_LAYER] head_size={head_size} num_heads={num_heads} "
                  f"num_kv_heads={self.num_kv_heads} sliding_window={self.sliding_window}",
                  flush=True)

        from vllm.model_executor.layers.quantization.kvarn.config import (
            KVarNConfig,
        )

        self.kvarn_config = KVarNConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Per-block fp16 tail buffer (in-progress tiles). Keyed by block_id.
        # Stage 3b uses a Python dict — small concurrent batch sizes only.
        # Stage 4 will move this into a dedicated GPU buffer.
        self._tails: dict[int, _BlockTail] = {}

        # Sink blocks (NEVER quantised, stay fp16 forever in self._tails).
        # Identified per-request as block_table[r][0]. Populated lazily during
        # ``forward()`` since ``do_kv_cache_update`` doesn't get block_table.
        # TODO(Stage 4.5.e): wire to vLLM's request-completion hook for eviction.
        self._sink_blocks: set[int] = set()

        # ── Stage α-2: deterministic per-block tail pool ─────────────────────
        # Each block_id maps to slot = block_id in the pool (no allocator,
        # no dict). Pool is sized to kv_cache.shape[0] = num_blocks at first
        # `_ensure_pool` call. Sink blocks stay in the pool permanently;
        # non-sink blocks have their slot's content quantised into the int4
        # cache at tile-boundary flushes (triggered from the metadata
        # builder, between captured graph replays).
        self._tail_K_pool: torch.Tensor | None = None   # [POOL_SIZE, group, Hk, D] fp16
        self._tail_V_pool: torch.Tensor | None = None
        self._draft_K_scratch: torch.Tensor | None = None  # [DRAFT_SLOTS, Hk, D] fp16
        self._draft_V_scratch: torch.Tensor | None = None
        # Per-instance shorthand views of the class-level per-device tensors
        # (so kernels can read without dict lookups). Re-bound on every
        # _ensure_pool call.
        self._is_sink_t: torch.Tensor | None = None        # [num_blocks] bool
        self._block_to_slot_t: torch.Tensor | None = None  # [num_blocks] int32
        self._block_lookup_size: int = 0

        # Cached fp16 Hadamard for the rotate-on-store matmul in
        # do_kv_cache_update (avoids a per-call .float() cast that allocates).
        self._H_fp16: torch.Tensor | None = None

        # Store-side rotation scratch (pre-allocated by _ensure_pool so the
        # captured forward never allocates). Shapes:
        #   _k_rot_scratch  [max_num_batched_tokens, Hk, D] fp16
        #   _v_rot_scratch  [max_num_batched_tokens, Hk, D] fp16
        self._k_rot_scratch: torch.Tensor | None = None
        self._v_rot_scratch: torch.Tensor | None = None

        # Reference to this layer's int4 kv_cache, captured on the first
        # forward(). The metadata builder uses it to drive tile-boundary
        # flushes into this layer's cache (outside the captured region).
        self._kv_cache_ref: torch.Tensor | None = None


        # Stage 5.a Step 7 — decode scratch. These instance attrs are
        # bound by `_ensure_pool` to per-device class-shared tensors so all
        # 28 attention layers reuse a single set of buffers.
        self._q_fp32_buf: torch.Tensor | None = None
        self._q_rot_fp32_buf: torch.Tensor | None = None
        self._q_rot_fp16_buf: torch.Tensor | None = None
        self._out_rot_fp32_buf: torch.Tensor | None = None
        self._output_fp32_buf: torch.Tensor | None = None
        self._fused_out_buf: torch.Tensor | None = None
        self._mid_o_buf: torch.Tensor | None = None
        self._mid_lse_buf: torch.Tensor | None = None
        self._fa_K_buf: torch.Tensor | None = None
        self._fa_V_buf: torch.Tensor | None = None
        self._prefill_out_buf: torch.Tensor | None = None
        self._materialized_q_rot_buf: torch.Tensor | None = None
        self._materialized_out_buf: torch.Tensor | None = None

        self._materialized_attn_backend = os.environ.get(
            "KVARN_MATERIALIZED_ATTN_BACKEND", "TRITON_ATTN"
        ).upper()
        if self._materialized_attn_backend in {"TRITON", "VLLM_TRITON"}:
            self._materialized_attn_backend = "TRITON_ATTN"
        elif self._materialized_attn_backend in {"FLASH", "FLASH_ATTN_VARLEN"}:
            self._materialized_attn_backend = "FLASH_ATTN"

        self.fa_version = get_flash_attn_version(head_size=head_size)

        # Look up serving caps so scratch can be sized once, generously
        # enough that capture probes don't trigger a resize inside the
        # captured region. Falls back to conservative defaults if the
        # global config isn't available (unit tests, etc).
        try:
            from vllm.config import get_current_vllm_config
            _cfg = get_current_vllm_config()
            self._max_num_seqs = _cfg.scheduler_config.max_num_seqs
            self._max_num_batched_tokens = _cfg.scheduler_config.max_num_batched_tokens
            self._max_model_len = _cfg.model_config.max_model_len
            self._num_hidden_layers = getattr(_cfg.model_config.hf_config,
                                              "num_hidden_layers", 32)
        except Exception:
            self._max_num_seqs = 256
            self._max_num_batched_tokens = 8192
            self._num_hidden_layers = 32
            self._max_model_len = 4096

        # Register so the metadata builder can find us (slot allocation /
        # sink marking / flush triggers all enumerate _all_impls).
        type(self)._all_impls.append(self)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ensure_pool(self, device: torch.device, num_blocks_hint: int = 0) -> None:
        """Lazy-allocate the GPU tail pool + lookup tensors + decode scratch.

        Stage α-2: pool is a *fixed-size* sparse buffer
        ([POOL_SIZE, group, Hk, D]). A class-level allocator maps block_id →
        slot (with a GPU lookup tensor `_block_to_slot_t_per_device[device]`
        sized to num_blocks). Pool size = ~2 × max_num_seqs (covers
        sink + in-progress tail for the largest captured batch).

        All allocation happens BEFORE the captured forward, so
        do_kv_cache_update can be pure tensor ops.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        cfg = self.kvarn_config
        cls = type(self)

        # Pool: fixed size, per-instance because each layer holds unique K/V.
        if self._tail_K_pool is None:
            # Peak live slots in a mixed prefill+decode step:
            #   • every active seq's sink + in-progress tail        → 2·max_num_seqs
            #   • every block written by chunked prefills this step → up to
            #     max_num_batched_tokens / group (full blocks not yet flushed)
            # The decode-only bound (2·max_num_seqs) underflows during bursty
            # serving where many requests prefill simultaneously → "pool
            # exhausted". Size for the mixed-batch peak.
            group = cfg.group
            prefill_blocks = (self._max_num_batched_tokens + group - 1) // group
            naive = 2 * self._max_num_seqs + prefill_blocks + 32

            # DECOUPLE pool size from max_num_seqs. Sizing the pool to 2·mns
            # over-provisions when actual concurrency is memory-bound below mns
            # (long context) — empirically that dropped 2-bit throughput 0.99x→
            # 0.88x FP16 and cut effective capacity 4.36x→2.75x at seq 8192.
            # Cap pool memory at a fraction of total GPU memory (default 8%);
            # the pool is per-layer fp16 (sink + tail), one slot =
            # group · num_kv_heads · head_dim · 4 bytes (K+V fp16). User can
            # pin exactly via KVARN_POOL_SLOTS for precise tuning.
            env_slots = int(os.environ.get("KVARN_POOL_SLOTS", "0"))
            if env_slots > 0:
                pool_size = max(env_slots, 64)
            else:
                frac = float(os.environ.get("KVARN_POOL_MEM_FRAC", "0.08"))
                total_bytes = torch.cuda.get_device_properties(device).total_memory
                slot_bytes_per_layer = group * self.num_kv_heads * cfg.head_dim * 4
                cap = int(total_bytes * frac
                          / (slot_bytes_per_layer * max(self._num_hidden_layers, 1)))
                pool_size = max(min(naive, max(cap, 64)), 64)
            self._tail_K_pool = torch.zeros(
                pool_size, cfg.group, self.num_kv_heads, cfg.head_dim,
                dtype=torch.float16, device=device,
            )
            self._tail_V_pool = torch.zeros_like(self._tail_K_pool)
        else:
            pool_size = self._tail_K_pool.shape[0]

        # Per-GROUP allocator state — ensure it exists for THIS group_key.
        # Decoupled from the per-impl pool allocation above: the impl's
        # _group_key is set to the proxy in __init__ and later RE-TAGGED to the
        # true (per-group) key by the builder, so the pool may already exist
        # under a stale key when this group_key is first seen. Idempotent.
        gk = self._group_key
        if gk not in cls._free_slots:
            cls._free_slots[gk] = list(range(pool_size - 1, -1, -1))
            cls._allocator_pool_size[gk] = pool_size
            cls._block_to_slot_dict[gk] = {}
            cls._global_sink_blocks[gk] = set()
        if gk not in cls._draft_free_indices:
            max_spec_tokens = max(int(os.environ.get("KVARN_MTP_MAX_DRAFTS", "8")), 1)
            draft_slots = int(os.environ.get("KVARN_DRAFT_SCRATCH_SLOTS", "0"))
            if draft_slots <= 0:
                draft_slots = max(self._max_num_seqs * max_spec_tokens + 32, 64)
            cls._draft_free_indices[gk] = list(range(draft_slots - 1, -1, -1))
            cls._draft_slot_to_index[gk] = {}
            cls._draft_scratch_size[gk] = draft_slots

        draft_slots = cls._draft_scratch_size[gk]
        if self._draft_K_scratch is None:
            self._draft_K_scratch = torch.empty(
                draft_slots, self.num_kv_heads, cfg.head_dim,
                dtype=torch.float16, device=device)
            self._draft_V_scratch = torch.empty_like(self._draft_K_scratch)

        # GPU lookup tensors, keyed by (device, group_key): each KV-cache group
        # has its own block_id space, so the two groups must NOT share a mirror.
        gk = self._group_key
        mkey = (device, gk)
        num_blocks = max(num_blocks_hint, cls._max_known_block_id.get(gk, 0) + 1, 1024)
        existing = cls._block_to_slot_t_per_device.get(mkey)
        if existing is None or existing.shape[0] < num_blocks:
            new_b2s = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
            new_is_sink = torch.zeros(num_blocks, dtype=torch.bool, device=device)
            # Re-sync from this group's CPU state (rare, only on resize / first init).
            for bid, slot in cls._block_to_slot_dict.get(gk, {}).items():
                if bid < num_blocks:
                    new_b2s[bid] = slot
            for bid in cls._global_sink_blocks.get(gk, set()):
                if bid < num_blocks:
                    new_is_sink[bid] = True
            cls._block_to_slot_t_per_device[mkey] = new_b2s
            cls._is_sink_t_per_device[mkey] = new_is_sink
        # Per-instance shorthand pointers so the decode driver / kernels read
        # without dict lookups in the hot path.
        self._is_sink_t = cls._is_sink_t_per_device[mkey]
        self._block_to_slot_t = cls._block_to_slot_t_per_device[mkey]
        self._block_lookup_size = self._block_to_slot_t.shape[0]

        # Cached fp16 Hadamard for the rotate-on-store matmul.
        if self._H_fp16 is None:
            self._H_fp16 = self._hadamard(device).to(torch.float16).contiguous()

        # One-time flush-kernel warmup (issue #15). The Sinkhorn + int4-store
        # kernels are exercised ONLY at a tile-boundary flush, which never
        # happens during vLLM's profiling/dummy run (no request crosses a block
        # boundary there). So they JIT-compile on the FIRST real flush DURING
        # serving — a multi-hundred-ms stall that surfaces as a latency spike
        # and a `jit_monitor` "JIT compilation during inference" warning, and
        # disproportionately hurts low-concurrency aggregate throughput (the
        # one-time cost lands inside a small measured window). Compile them here,
        # once per shape/config, at pool-init time (outside any captured region)
        # using the exact tile shapes the flush uses, so serving never pays it.
        warm_key = (device, cfg.head_dim, cfg.group, cfg.key_bits, cfg.value_bits)
        if warm_key not in cls._kernel_warmed:
            k_dummy = torch.zeros(
                1, cfg.head_dim, cfg.group, dtype=torch.float16, device=device)
            v_dummy = torch.zeros(
                1, cfg.group, cfg.head_dim, dtype=torch.float16, device=device)
            _sinkhorn_pack_kv(k_dummy, v_dummy, cfg)
            cls._kernel_warmed.add(warm_key)

        # Decode-kernel warmup (issue #10). The DECODE kernels (fused
        # single-stage incl. its @triton.autotune sweep, split-K stage1/2, and
        # the packed-KV build kernel) never run during vLLM's prefill-shaped
        # profiling, so their one-time JIT + autotune cost (including the
        # autotuner's benchmark scratch) used to land in the FIRST real decode —
        # which, since v0.21, is the CUDA-graph memory estimation warmup. The
        # estimate then absorbed those one-time costs and over-charged "graph
        # memory" by GiBs, directly shrinking the derived KV-cache capacity.
        # Warm them here (profile time) on tiny synthetic state instead: the
        # cost is charged once to the memory profile, and the graph estimate
        # measures only real graph-pool memory. Keyed per (device, shape combo).
        dec_key = ("decode", device, cfg.head_dim, cfg.group, cfg.key_bits,
                   cfg.value_bits, self.num_heads, self.num_kv_heads,
                   int(getattr(self, "sliding_window", 0) or 0))
        if (dec_key not in cls._kernel_warmed
                and os.environ.get("KVARN_SKIP_DECODE_WARMUP", "0") != "1"):
            self._warm_decode_kernels(device)
            cls._kernel_warmed.add(dec_key)

        # Store-side rotation scratch.
        if self._k_rot_scratch is None:
            q_rows = max(self._max_num_batched_tokens, 1)
            self._k_rot_scratch = torch.empty(
                q_rows, self.num_kv_heads, cfg.head_dim,
                dtype=torch.float16, device=device,
            )
            self._v_rot_scratch = torch.empty_like(self._k_rot_scratch)
        # Decode scratch sized from vllm_config, SHARED across all impl
        # instances on this device (one set per device).
        D = cfg.head_dim
        Hq = self.num_heads
        Hk = self.num_kv_heads
        # Decode scratch rows. The decode driver indexes these buffers by
        # N = B * Hq (decode batch * query heads), so they must hold the largest
        # decode N as well as any prefill token count. A decode step (incl. its
        # CUDA-graph capture batch) has at most max_num_seqs queries, so the
        # decode bound is max_num_seqs * Hq. Sizing to the max of that and
        # max_num_batched_tokens makes the buffers correct for ANY
        # max_num_batched_tokens (the old code silently assumed
        # max_num_batched_tokens >= max_num_seqs * Hq, which breaks when it is
        # set low — e.g. a small chunked-prefill budget on a wide model).
        q_rows = max(self._max_num_batched_tokens, self._max_num_seqs * Hq, 1)
        # Materialized packed K/V scratch holds the total KV tokens attended in ONE
        # decode step (= sum of the batch's context lengths). The theoretical
        # bound max_num_seqs * max_model_len is pathological (e.g. 256×8192 =
        # 2.1M tokens ≈ 8.6 GB) and would starve the actual KV cache. Cap it at
        # KVARN_MATERIALIZED_KV_SCRATCH_TOKENS — enough for typical serving
        # and for the bench (single request up to max_model_len). The scratch
        # is per-step, shared across all layers, allocated ONCE.
        scratch_cap = int(os.environ.get(
            "KVARN_MATERIALIZED_KV_SCRATCH_TOKENS", "262144"))
        if scratch_cap <= 0:
            raise ValueError(
                "KVARN_MATERIALIZED_KV_SCRATCH_TOKENS must be positive.")
        fa_rows = max(min(self._max_num_seqs * self._max_model_len,
                          scratch_cap),
                      self._max_model_len, 4096)
        materialized_rows = max(self._max_num_batched_tokens * Hq,
                                self._max_num_seqs * Hq, 1)
        cls = type(self)
        # Key the shared decode scratch by (device, D, Hk), NOT device alone:
        # heterogeneous-head models (e.g. Gemma-4: 256-dim/16-kv sliding layers +
        # 512-dim/4-kv global layers) have multiple (head_dim, kv_heads) combos,
        # and a buffer sized for one combo's D/Hk is the wrong width for another
        # (caused a reshape(N,512)-on-256-wide-buffer crash). One scratch set per
        # combo (Gemma-4 = 2 sets; cost is small).
        bkey = (device, D, Hk)
        if bkey not in cls._shared_q_fp32_buf:
            cls._shared_q_fp32_buf[bkey] = torch.empty(q_rows, D, dtype=torch.float32, device=device)
            cls._shared_q_rot_fp32_buf[bkey] = torch.empty(q_rows, D, dtype=torch.float32, device=device)
            cls._shared_q_rot_fp16_buf[bkey] = torch.empty(q_rows, D, dtype=torch.float16, device=device)
            cls._shared_out_rot_fp32_buf[bkey] = torch.empty(q_rows, D, dtype=torch.float32, device=device)
            cls._shared_output_fp32_buf[bkey] = torch.empty(q_rows, D, dtype=torch.float32, device=device)
            cls._shared_fused_out_buf[bkey] = torch.empty(q_rows, D, dtype=torch.float16, device=device)
            cls._shared_prefill_out_buf[bkey] = torch.empty(q_rows, Hq, D, dtype=torch.float16, device=device)
        _ex_mq = cls._shared_materialized_q_rot_buf.get(bkey)
        if _ex_mq is None or _ex_mq.shape[0] < materialized_rows:
            cls._shared_materialized_q_rot_buf[bkey] = torch.empty(
                materialized_rows, D, dtype=torch.float16, device=device)
            cls._shared_materialized_out_buf[bkey] = torch.empty(
                materialized_rows, D, dtype=torch.float16, device=device)
        from vllm.v1.attention.ops.triton_kvarn_decode import adaptive_num_kv_splits
        # Split-K partial buffers, sized to EXACTLY what the split-K decode path
        # can index: it runs ONLY on pure single-query decode steps, whose row
        # count is N = B*Hq with B <= max_num_seqs — NOT q_rows (which is
        # max_num_batched_tokens-driven and sized the buffer ~85x too big at
        # typical configs: 256 MiB instead of ~3 MiB for max_num_seqs=2/Hq=24/
        # 64 splits; issue #10 follow-up). Split count matches the driver's
        # adaptive helper (same max_model_len) or a larger adaptive count would
        # overflow a smaller buffer; the driver additionally falls back to the
        # single-stage kernel if N ever exceeds the buffer rows (defensive —
        # e.g. an oversized padded dummy batch).
        _splits = adaptive_num_kv_splits((self._max_model_len + cfg.group - 1) // cfg.group)
        # Rows are bounded by the split-K REGIME, not max_num_seqs: the driver
        # only takes split-K when B*Hk <= sm_count (otherwise the single-stage
        # kernel runs), so the most rows it can ever index is (sm_count//Hk)*Hq
        # = sm_count*Q_PER_KV, independent of max_num_seqs. Sizing to
        # max_num_seqs*Hq over-reserved this fp32 partial buffer ~10-20x at high
        # concurrency, where it competes directly with the int4 KV cache. The
        # driver still falls back to single-stage if N ever exceeds these rows
        # (defensive), and split-K is never disabled for a batch it would take
        # (B*Hk<=sm_count => N=B*Hq <= (sm_count//Hk)*Hq).
        _sm = (getattr(self, "_sm_count", 0)
               or torch.cuda.get_device_properties(device).multi_processor_count)
        mid_rows = max((_sm // max(Hk, 1)) * Hq, Hq, 1)
        _ex_mid = cls._shared_mid_o_buf.get(bkey)
        if _ex_mid is None or _ex_mid.shape[0] < mid_rows or _ex_mid.shape[1] != _splits:
            cls._shared_mid_o_buf[bkey] = torch.empty(mid_rows, _splits, D, dtype=torch.float32, device=device)
            cls._shared_mid_lse_buf[bkey] = torch.empty(mid_rows, _splits, dtype=torch.float32, device=device)
        if bkey not in cls._shared_fa_K_buf or cls._shared_fa_K_buf[bkey].shape[0] < fa_rows:
            cls._shared_fa_K_buf[bkey] = torch.zeros(fa_rows, Hk, D, dtype=torch.float16, device=device)
            cls._shared_fa_V_buf[bkey] = torch.zeros_like(cls._shared_fa_K_buf[bkey])
        # Mirror to instance attrs for fast access by the decode driver.
        self._q_fp32_buf = cls._shared_q_fp32_buf[bkey]
        self._q_rot_fp32_buf = cls._shared_q_rot_fp32_buf[bkey]
        self._q_rot_fp16_buf = cls._shared_q_rot_fp16_buf[bkey]
        self._out_rot_fp32_buf = cls._shared_out_rot_fp32_buf[bkey]
        self._output_fp32_buf = cls._shared_output_fp32_buf[bkey]
        self._fused_out_buf = cls._shared_fused_out_buf[bkey]
        self._mid_o_buf = cls._shared_mid_o_buf[bkey]
        self._mid_lse_buf = cls._shared_mid_lse_buf[bkey]
        self._fa_K_buf = cls._shared_fa_K_buf[bkey]
        self._fa_V_buf = cls._shared_fa_V_buf[bkey]
        self._prefill_out_buf = cls._shared_prefill_out_buf[bkey]
        self._materialized_q_rot_buf = cls._shared_materialized_q_rot_buf[bkey]
        self._materialized_out_buf = cls._shared_materialized_out_buf[bkey]
    def _warm_decode_kernels(self, device: torch.device) -> None:
        """Compile + autotune every decode-path Triton kernel on tiny synthetic
        state (see the issue #10 note at the call site in ``_ensure_pool``).
        Uses throwaway tensors only — never touches the real cache/pool."""
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_build_packed_kv_kernel,
            _kvarn_fused_decode_kernel,
            _kvarn_fused_decode_stage1,
            _kvarn_fused_decode_stage2,
            adaptive_num_kv_splits,
        )

        cfg = self.kvarn_config
        D, G = cfg.head_dim, cfg.group
        Hq, Hk = self.num_heads, self.num_kv_heads
        B, n_blocks = 8, 4
        sw = int(getattr(self, "sliding_window", 0) or 0)

        cache = torch.zeros(B * n_blocks, Hk, cfg.tile_bytes_aligned,
                            dtype=torch.uint8, device=device)
        pool_k = torch.zeros(1, G, Hk, D, dtype=torch.float16, device=device)
        pool_v = torch.zeros_like(pool_k)
        b2s = torch.full((B * n_blocks,), -1, dtype=torch.int32, device=device)
        bt = torch.arange(B * n_blocks, dtype=torch.int32,
                          device=device).view(B, n_blocks)
        sl = torch.full((B,), n_blocks * G, dtype=torch.int32, device=device)
        q = torch.zeros(B, Hq, D, dtype=torch.float16, device=device)
        out = torch.zeros_like(q)

        qpk = Hq // Hk
        qpk_pad = 1 << (qpk - 1).bit_length() if qpk > 1 else 1
        common = dict(
            MAX_BLOCKS_PER_REQ=n_blocks, D=D, GROUP=G,
            Q_PER_KV=qpk, Q_PER_KV_PAD=qpk_pad, SLIDING_WINDOW=sw,
            K_BITS=cfg.key_bits, V_BITS=cfg.value_bits,
            NUM_BLOCKS_LOOKUP=B * n_blocks,
            K_PACKED_OFFSET=cfg.k_packed_offset, K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset, K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset, V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset, V_ZP_OFFSET=cfg.v_zp_offset,
            VQ_INDIRECT=False,
        )
        # 1. Single-stage fused kernel — runs the @triton.autotune sweep.
        # (sl doubles as the unused Req_row_ptr dummy; see VQ_INDIRECT.)
        _kvarn_fused_decode_kernel[(B, Hk)](
            q, sl, bt, sl, b2s, cache, pool_k, pool_v, out, self.scale,
            Hq * D, D, bt.stride(0), cache.stride(0), cache.stride(1),
            pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
            Hq * D, D, **common,
        )
        # 2. Split-K stage1 + stage2, with the exact split count and launch
        # knobs the decode driver will use for this deployment.
        splits = adaptive_num_kv_splits((self._max_model_len + G - 1) // G)
        mid_o = torch.zeros(B * Hq, splits, D, dtype=torch.float32, device=device)
        mid_lse = torch.zeros(B * Hq, splits, dtype=torch.float32, device=device)
        # stage1 is @triton.autotune'd; this warmup launch triggers its sweep
        # here (pre-CUDA-graph-capture) so capture never benchmarks.
        _kvarn_fused_decode_stage1[(B, Hk, splits)](
            q, sl, bt, sl, b2s, cache, pool_k, pool_v, mid_o, mid_lse, self.scale,
            Hq * D, D, bt.stride(0), cache.stride(0), cache.stride(1),
            pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
            mid_o.stride(0), mid_o.stride(1), mid_lse.stride(0),
            NUM_KV_SPLITS=splits, HQ=Hq, **common,
        )
        out2d = out.view(B * Hq, D)
        _kvarn_fused_decode_stage2[(B * Hq,)](
            mid_o, mid_lse, out2d,
            mid_o.stride(0), mid_o.stride(1), mid_lse.stride(0),
            out2d.stride(0), D=D, NUM_KV_SPLITS=splits, num_warps=2,
        )
        # 2b. VQ_INDIRECT (fused spec-verify) specializations — separate
        # compiled variants; warm them so the FIRST MTP verify step doesn't
        # pay the Triton JIT mid-serving.
        vq_rows = torch.zeros(B, dtype=torch.int32, device=device)
        common_vq = dict(common, VQ_INDIRECT=True)
        _kvarn_fused_decode_kernel[(B, Hk)](
            q, vq_rows, bt, sl, b2s, cache, pool_k, pool_v, out, self.scale,
            Hq * D, D, bt.stride(0), cache.stride(0), cache.stride(1),
            pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
            Hq * D, D, **common_vq,
        )
        _kvarn_fused_decode_stage1[(B, Hk, splits)](
            q, vq_rows, bt, sl, b2s, cache, pool_k, pool_v, mid_o, mid_lse,
            self.scale,
            Hq * D, D, bt.stride(0), cache.stride(0), cache.stride(1),
            pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
            mid_o.stride(0), mid_o.stride(1), mid_lse.stride(0),
            NUM_KV_SPLITS=splits, HQ=Hq, **common_vq,
        )
        # 2c. Shared-dequant verify kernel (uniform-QLEN spec verify) — runs
        # its @triton.autotune sweep here so capture never benchmarks. QLEN is
        # the deployment's 1 + num_speculative_tokens.
        try:
            from vllm.config import get_current_vllm_config
            _spec = get_current_vllm_config().speculative_config
            _qlen = 1 + int(_spec.num_speculative_tokens) if _spec else 0
        except Exception:
            _qlen = 0
        if _qlen >= 2:
            from vllm.v1.attention.ops.triton_kvarn_decode import (
                _kvarn_fused_verify_stage1,
            )
            nq = B * _qlen
            sl_vq = sl.repeat_interleave(_qlen)
            qv = torch.zeros(nq, Hq, D, dtype=torch.float16, device=device)
            mid_o_v = torch.zeros(nq * Hq, splits, D, dtype=torch.float32,
                                  device=device)
            mid_lse_v = torch.zeros(nq * Hq, splits, dtype=torch.float32,
                                    device=device)
            common_v = dict(common)
            _kvarn_fused_verify_stage1[(B, Hk, splits)](
                qv, bt, sl_vq, b2s, cache, pool_k, pool_v,
                mid_o_v, mid_lse_v, self.scale,
                Hq * D, D, bt.stride(0), cache.stride(0), cache.stride(1),
                pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
                mid_o_v.stride(0), mid_o_v.stride(1), mid_lse_v.stride(0),
                QLEN=_qlen, HQ=Hq, NUM_KV_SPLITS=splits, **common_v,
            )
        # 3. Packed-KV build kernel (materialized cached-multiquery
        # spec-verify path).
        kp = torch.zeros(B * n_blocks * G, Hk, D, dtype=torch.float16, device=device)
        vp = torch.zeros_like(kp)
        cu_k = torch.arange(B + 1, dtype=torch.int32, device=device) * (n_blocks * G)
        _kvarn_build_packed_kv_kernel[(B * n_blocks, Hk)](
            bt, sl, cu_k, b2s, cache, pool_k, pool_v, kp, vp,
            bt.stride(0), cache.stride(0), cache.stride(1),
            pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
            kp.stride(0), kp.stride(1),
            MAX_BLOCKS_PER_REQ=n_blocks, D=D, GROUP=G,
            K_BITS=cfg.key_bits, V_BITS=cfg.value_bits,
            NUM_BLOCKS_LOOKUP=B * n_blocks,
            K_PACKED_OFFSET=cfg.k_packed_offset, K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset, K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset, V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset, V_ZP_OFFSET=cfg.v_zp_offset,
            num_warps=4, num_stages=2,
        )
        torch.cuda.synchronize(device)

    def _current_kvarn_metadata(self) -> KVarNMetadata | None:
        """Return this layer's KVarN metadata from the current forward context."""
        try:
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
        except Exception:
            return None
        md = getattr(ctx, "attn_metadata", None)
        if md is None:
            return None
        layer_name = getattr(self, "layer_name", None)
        if isinstance(md, dict):
            if layer_name is not None and isinstance(md.get(layer_name), KVarNMetadata):
                return md[layer_name]
            for m in md.values():
                if isinstance(m, KVarNMetadata):
                    return m
            return None
        if isinstance(md, list):
            for entry in md:
                if isinstance(entry, dict):
                    if (layer_name is not None
                            and isinstance(entry.get(layer_name), KVarNMetadata)):
                        return entry[layer_name]
                    for m in entry.values():
                        if isinstance(m, KVarNMetadata):
                            return m
            return None
        return md if isinstance(md, KVarNMetadata) else None

    def _hadamard(self, device: torch.device) -> torch.Tensor:
        return _build_hadamard(self.head_size, device)

    def _flat_block(self, kv_cache: torch.Tensor, block_id: int, head: int) -> torch.Tensor:
        """Contiguous ``[tile_bytes_aligned]`` uint8 view for one (block, head).

        ``kv_cache`` has shape ``(num_blocks, num_kv_heads, tile_bytes_aligned)``,
        so this selects a single contiguous row — no copy, writes propagate
        back to the cache tensor.
        """
        return kv_cache[block_id, head]

    def _write_packed(
        self, kv_cache: torch.Tensor, block_id: int, head: int,
        store_K: dict[str, torch.Tensor], store_V: dict[str, torch.Tensor],
    ) -> None:
        cfg = self.kvarn_config
        flat = self._flat_block(kv_cache, block_id, head)

        # K packed bytes
        ko = cfg.k_packed_offset
        flat[ko:ko + cfg.k_packed_bytes] = store_K["q_packed_uint8"].reshape(-1).to(torch.uint8)
        # K s_col, zp (per-channel, length D, fp16)
        flat[cfg.k_s_col_offset:cfg.k_s_col_offset + cfg.head_dim * 2].view(
            torch.float16
        )[:] = store_K["s_col_K"]
        flat[cfg.k_zp_offset:cfg.k_zp_offset + cfg.head_dim * 2].view(
            torch.float16
        )[:] = store_K["zp_K"]
        flat[cfg.k_s_row_offset:cfg.k_s_row_offset + cfg.group * 2].view(
            torch.float16
        )[:] = store_K["s_row_K"]

        # V packed bytes
        vo = cfg.v_packed_offset
        flat[vo:vo + cfg.v_packed_bytes] = store_V["q_packed_uint8"].reshape(-1).to(torch.uint8)
        flat[cfg.v_s_col_offset:cfg.v_s_col_offset + cfg.head_dim * 2].view(
            torch.float16
        )[:] = store_V["s_col_V"]
        flat[cfg.v_s_row_offset:cfg.v_s_row_offset + cfg.group * 2].view(
            torch.float16
        )[:] = store_V["s_row_V"]
        flat[cfg.v_zp_offset:cfg.v_zp_offset + cfg.group * 2].view(
            torch.float16
        )[:] = store_V["zp_V"]

    def _flush_tail(self, block_id: int, kv_cache: torch.Tensor) -> None:
        """Quantize a fully-filled tail buffer and write it into the cache.

        Stage α-2 sparse pool: pool slot = _block_to_slot_dict[block_id].
        Data is already rotated (rotation happens at do_kv_cache_update),
        so no `@ H` step here.
        """
        cfg = self.kvarn_config
        cls = type(self)
        slot = cls._block_to_slot_dict.get(self._group_key, {}).get(block_id)
        if slot is None:
            # Block has no pool slot — nothing to flush.
            self._tails.pop(block_id, None)
            return
        K_rot = self._tail_K_pool[slot].float()                   # [group, Hk, D]
        V_rot = self._tail_V_pool[slot].float()                   # [group, Hk, D]
        self._tails.pop(block_id, None)                           # drop tracker entry

        # Build batched per-head tiles (rows = absorb axis for each)
        K_tiles = K_rot.permute(1, 2, 0).contiguous()             # [Hk, D, group]
        V_tiles = V_rot.permute(1, 0, 2).contiguous()             # [Hk, group, D]

        # Sinkhorn + pack (fused launch when square head_dim==group, else
        # separate K/V launches — see _sinkhorn_pack_kv).
        K_out, V_out = _sinkhorn_pack_kv(K_tiles, V_tiles, cfg)
        Hk = self.num_kv_heads

        for h in range(Hk):
            store_K = {k: v[h] for k, v in K_out.items()}
            store_V = {k: v[h] for k, v in V_out.items()}
            self._write_packed(kv_cache, block_id, h, store_K, store_V)

        # NOTE: the pool slot is NOT freed here. The slot index addresses the
        # SAME row in every layer's pool, so it must stay allocated until ALL
        # layers have flushed their data into int4. The builder frees it once,
        # after iterating every impl (see "Free the flushed blocks' slots" in
        # build()). Freeing here would let layer 0's flush drop the slot, after
        # which layers 1..N find no slot (`.get()` → None) and silently skip
        # writing their int4 — corrupting all-but-the-first layer's history.

    @classmethod
    def _batched_flush(cls, flush_pairs: list) -> None:
        """Flush many (impl, block_id, kv_cache) tiles to int4.

        Dispatches to the vectorized path (default) or the legacy per-tile path
        (KVARN_FAST_FLUSH=0, kept for A/B + the tile-dump debug hook). The
        vectorized path replaces the per-(layer,block,head) Python gather/write
        loops — which exploded into ~10^5 tiny GPU ops on a synchronized burst
        (prefill completion, lockstep decode boundary) and dominated build() at
        high concurrency (issue #15: ~44 ms/step at B=256) — with one
        index_select gather + one index_copy write per (layer, block-chunk).
        Numerically identical: same Sinkhorn, same RTN/pack math, same byte
        layout; only the data movement is batched."""
        if not flush_pairs:
            return
        if os.environ.get("KVARN_FAST_FLUSH", "1") != "1":
            return cls._batched_flush_legacy(flush_pairs)

        cfg = flush_pairs[0][0].kvarn_config
        Hk = flush_pairs[0][0].num_kv_heads
        D = cfg.head_dim
        G = cfg.group
        T = cfg.tile_bytes_aligned
        kpb = cfg.k_packed_bytes
        vpb = cfg.v_packed_bytes

        # Group by impl (layer); every impl flushes the SAME block set (the
        # builder cross-products flush_block_ids with group_impls) and pool slot
        # indices are shared across layers, so per impl we have (kvc, bids, slots).
        by_impl: dict = {}
        for impl, bid, kvc in flush_pairs:
            slot = cls._block_to_slot_dict.get(impl._group_key, {}).get(bid)
            if slot is None:
                impl._tails.pop(bid, None)
                continue
            e = by_impl.get(id(impl))
            if e is None:
                e = [impl, kvc, [], []]
                by_impl[id(impl)] = e
            e[2].append(bid)
            e[3].append(slot)
            impl._tails.pop(bid, None)
        if not by_impl:
            return

        # Block-chunk so one Sinkhorn launch stays bounded (~2k [R,C] tiles).
        CHUNK_BLOCKS = max(1, 2048 // max(Hk, 1))
        for impl, kvc, bids, slots in by_impl.values():
            if kvc is None:
                continue
            dev = impl._tail_K_pool.device
            # WSL fix (PR #16): one H2D for the whole block set, slice on device
            # per chunk (a torch.as_tensor H2D per chunk is a sync, ~100x on WSL).
            slots_dev = torch.as_tensor(slots, dtype=torch.long, device=dev)
            bids_dev = torch.as_tensor(bids, dtype=torch.long, device=dev)
            for c0 in range(0, len(bids), CHUNK_BLOCKS):
                bchunk = bids[c0:c0 + CHUNK_BLOCKS]
                nB = len(bchunk)
                slot_t = slots_dev[c0:c0 + CHUNK_BLOCKS]
                bid_t = bids_dev[c0:c0 + CHUNK_BLOCKS]
                # One gather per chunk (was nB tiny .float() ops).
                K_rot = impl._tail_K_pool.index_select(0, slot_t).float()  # [nB,G,Hk,D]
                V_rot = impl._tail_V_pool.index_select(0, slot_t).float()
                # Tiles: K [N, D, G] (absorb=channel), V [N, G, D] (absorb=token).
                K_tiles = K_rot.permute(0, 2, 3, 1).reshape(nB * Hk, D, G)
                V_tiles = V_rot.permute(0, 2, 1, 3).reshape(nB * Hk, G, D)
                K_out, V_out = _sinkhorn_pack_kv(K_tiles, V_tiles, cfg)
                # Assemble the packed cache record [nB*Hk, tile_bytes] by
                # concatenating fields in config-offset order (fp16 scales
                # byte-reinterpreted to uint8), then pad to tile_bytes_aligned.
                M = nB * Hk
                parts = [
                    K_out["q_packed_uint8"].reshape(M, kpb),
                    K_out["s_col_K"].contiguous().view(torch.uint8),
                    K_out["zp_K"].contiguous().view(torch.uint8),
                    K_out["s_row_K"].contiguous().view(torch.uint8),
                    V_out["q_packed_uint8"].reshape(M, vpb),
                    V_out["s_col_V"].contiguous().view(torch.uint8),
                    V_out["s_row_V"].contiguous().view(torch.uint8),
                    V_out["zp_V"].contiguous().view(torch.uint8),
                ]
                rec = torch.cat(parts, dim=1)                       # [M, tile_bytes]
                if rec.shape[1] < T:
                    rec = torch.nn.functional.pad(rec, (0, T - rec.shape[1]))
                # One scatter per chunk (was nB*Hk _write_packed calls).
                kvc[bid_t] = rec.view(nB, Hk, T)

    @classmethod
    def _batched_flush_legacy(cls, flush_pairs: list) -> None:
        """Flush many (impl, block_id, kv_cache) tiles via batched Sinkhorn + RTN.

        Replaces the per-(layer, block) Python loop calling `_flush_tail`. At
        burst with many layers × many lockstep boundary crossings, the per-call
        kernel-launch + Python-iter overhead dominated; Sinkhorn and the RTN-
        pack are per-tile-independent, so stacking is numerically identical
        (no accuracy change).

        Chunked at CHUNK_PAIRS to bound the transient gather memory — at peak
        (48 layers × ~73 lockstep reqs = ~3.5k pairs), the unchunked stack hits
        >2 GB of fp32 working memory and OOMs on a memory-tight burst.
        """
        if not flush_pairs:
            return
        CHUNK_PAIRS = 256
        cfg = flush_pairs[0][0].kvarn_config
        Hk = flush_pairs[0][0].num_kv_heads
        # Pre-filter pairs that still have a pool slot (some may have been freed
        # by a sibling impl's flush already during this builder call).
        filt: list[tuple] = []
        for impl, bid, kvc in flush_pairs:
            slot = cls._block_to_slot_dict.get(impl._group_key, {}).get(bid)
            if slot is None:
                impl._tails.pop(bid, None)
                continue
            filt.append((impl, bid, kvc, slot))
            impl._tails.pop(bid, None)
        if not filt:
            return
        for c0 in range(0, len(filt), CHUNK_PAIRS):
            chunk = filt[c0:c0 + CHUNK_PAIRS]
            N = len(chunk)
            # Gather pool data for this chunk.
            K_list = [impl._tail_K_pool[slot].float() for impl, _, _, slot in chunk]   # [G, Hk, D]
            V_list = [impl._tail_V_pool[slot].float() for impl, _, _, slot in chunk]
            K_stack = torch.stack(K_list, dim=0)                                       # [N, G, Hk, D]
            V_stack = torch.stack(V_list, dim=0)
            # Optional: dump first chunk's raw (pre-Sinkhorn) tiles for outlier
            # analysis (KVARN_DUMP_TILES=/path/to/file.pt).
            dump_path = os.environ.get("KVARN_DUMP_TILES", "")
            if dump_path and not getattr(cls, "_tiles_dumped", False):
                cls._tiles_dumped = True
                # Capture per-tile (layer_idx, block_id) for per-layer analysis.
                # layer_idx pulled from impl.layer_name (e.g. "model.layers.7.self_attn")
                # via a regex fallback to enumerate index if name parsing fails.
                import re
                lyr_ids, blk_ids = [], []
                for impl, bid, _, _ in chunk:
                    name = getattr(impl, "layer_name", "") or ""
                    m = re.search(r"layers\.(\d+)\b", name)
                    lyr_ids.append(int(m.group(1)) if m else -1)
                    blk_ids.append(int(bid))
                torch.save({"K_stack": K_stack.detach().cpu(),
                            "V_stack": V_stack.detach().cpu(),
                            "layer_ids": lyr_ids,
                            "block_ids": blk_ids,
                            "Hk": flush_pairs[0][0].num_kv_heads,
                            "G": cfg.group, "D": cfg.head_dim,
                            "key_bits": cfg.key_bits, "value_bits": cfg.value_bits,
                            "sinkhorn_iters": cfg.sinkhorn_iters},
                           dump_path)
                print(f"[KVARN] dumped {N} (layer,block) pre-Sinkhorn tiles → {dump_path}",
                      flush=True)
                print(f"[KVARN] layer_ids in dump: {sorted(set(lyr_ids))}", flush=True)
            del K_list, V_list
            # K tile per Sinkhorn batch row: [D, G] (absorb = channel).
            K_tiles = K_stack.permute(0, 2, 3, 1).reshape(N * Hk, K_stack.shape[3], K_stack.shape[1])
            V_tiles = V_stack.permute(0, 2, 1, 3).reshape(N * Hk, V_stack.shape[1], V_stack.shape[3])
            del K_stack, V_stack
            # Sinkhorn + pack (fused when square head_dim==group, else separate
            # K/V launches for non-square head_dim=256 — see _sinkhorn_pack_kv).
            K_out, V_out = _sinkhorn_pack_kv(K_tiles, V_tiles, cfg)
            del K_tiles, V_tiles
            # Distribute packed results to each (layer, block, head) cache slot.
            for i, (impl, bid, kvc, _) in enumerate(chunk):
                for h in range(Hk):
                    idx = i * Hk + h
                    store_K = {k: v[idx] for k, v in K_out.items()}
                    store_V = {k: v[idx] for k, v in V_out.items()}
                    impl._write_packed(kvc, bid, h, store_K, store_V)
            del K_out, V_out

    # ── do_kv_cache_update ───────────────────────────────────────────────────

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Append incoming tokens to the per-block fp16 tail buffers.

        We DO NOT flush here. Flushing requires the block_table (to know
        which block IDs are "sink" blocks for each request), which is only
        available via ``attn_metadata`` in ``forward()``. The flush is
        therefore deferred to ``_flush_eligible_tails``, invoked at the top
        of ``forward()``.
        """
        # Stage α-2: fully tensorised store. rotate(k, v) by H_fp16 → scatter
        # into pool at slot=block_id directly. No Python loop, no allocator,
        # no dict mutation. Safe inside a captured CUDA graph.
        cfg = self.kvarn_config
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        device = key.device
        Hk = self.num_kv_heads
        D = self.head_size

        # bf16 boundary-cast (see forward): KVarN store/rotation is fp16.
        if key.dtype != torch.float16:
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        # Ensure pool + lookup tensors + rotation scratch exist (no-op during
        # capture; first call before capture sizes pool to kv_cache num_blocks).
        self._ensure_pool(device, num_blocks_hint=kv_cache.shape[0])
        md = self._current_kvarn_metadata()
        store_slot_mapping = slot_mapping[:N]
        if (md is not None and md.store_slot_mapping is not None
                and md.store_slot_mapping.shape[0] >= N):
            store_slot_mapping = md.store_slot_mapping[:N]

        # Reshape to (N, Hk, D) — view, no copy (key/value already fp16).
        k_view = key[:N].view(N, Hk, D)
        v_view = value[:N].view(N, Hk, D)

        # Rotate via cached fp16 Hadamard. torch.matmul `out=` is
        # capture-friendly (uses the caching allocator's pool).
        k_rot = self._k_rot_scratch[:N]
        v_rot = self._v_rot_scratch[:N]
        torch.matmul(k_view, self._H_fp16, out=k_rot)
        torch.matmul(v_view, self._H_fp16, out=v_rot)

        # Scatter via the sparse pool indirection. Slot lookup (block_id →
        # pool slot) is done inside the kernel against the GPU
        # _block_to_slot_t tensor (mutated only by the metadata builder).
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_scatter_draft_store_kernel,
            _kvarn_scatter_store_kernel,
        )
        _kvarn_scatter_store_kernel[(N, Hk)](
            k_rot, v_rot, store_slot_mapping,
            self._block_to_slot_t,
            self._tail_K_pool, self._tail_V_pool,
            k_rot.stride(0), k_rot.stride(1),
            self._tail_K_pool.stride(0),
            self._tail_K_pool.stride(1),
            self._tail_K_pool.stride(2),
            GROUP=cfg.group, D=D,
            NUM_BLOCKS_LOOKUP=self._block_lookup_size,
            num_warps=2, num_stages=2,
        )
        if (md is not None and md.has_transient_query_kv
                and md.draft_store_indices is not None
                and md.draft_store_indices.shape[0] >= N):
            if self._draft_K_scratch is None or self._draft_V_scratch is None:
                raise RuntimeError("KVarN draft scratch was not initialized.")
            _kvarn_scatter_draft_store_kernel[(N, Hk)](
                k_rot, v_rot, md.draft_store_indices[:N],
                self._draft_K_scratch, self._draft_V_scratch,
                k_rot.stride(0), k_rot.stride(1),
                self._draft_K_scratch.stride(0),
                self._draft_K_scratch.stride(1),
                D=D,
                num_warps=2, num_stages=2,
            )
        # No CPU bookkeeping here — fill tracking + flush triggering live in
        # KVarNMetadataBuilder.build() (outside the captured region). This
        # method is now pure tensor ops, safe inside a captured CUDA graph.

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "KVarNMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        device = query.device

        if output is None:
            output = torch.zeros(
                num_tokens, self.num_heads * self.head_size,
                dtype=query.dtype, device=device,
            )
        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        # Make sure pool + block-lookup tensors exist and cover num_blocks.
        self._ensure_pool(kv_cache.device, num_blocks_hint=kv_cache.shape[0])
        # Cache the kv_cache ref so the metadata builder can drive flushes
        # into this layer's int4 cache (outside the captured region).
        self._kv_cache_ref = kv_cache

        # Flush is now triggered from KVarNMetadataBuilder.build() between
        # captured graph replays — nothing to do here at the top of forward.

        # bf16 boundary-cast: KVarN's compute (rotation matmul, scratch buffers,
        # Triton stores) is fp16 internally. Cast bf16 activations to fp16 at this
        # entry point; the output write below casts back to output.dtype. fp16 is
        # untouched (byte-identical), and the cast is lossless for KVarN (fp16
        # mantissa > bf16, and the cache is 4-bit). Without this, bf16 q mixing
        # with fp16 KV buffers trips "Expected out BFloat16, got Half".
        if query.dtype != torch.float16:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        q = query[:N].view(N, self.num_heads, self.head_size)

        if not attn_metadata.is_prefill:
            attn_out = self._decode_path(q, kv_cache, attn_metadata)
        elif (attn_metadata.vq_seqlen is not None
              and attn_metadata.num_decode_tokens == N):
            # Pure multi-token decode batch = a spec-decode verify step
            # (uniform query length under graph capture). One fused-kernel
            # pass over the vq plan — fully graph-capturable.
            if attn_metadata.has_transient_query_kv:
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._cached_multiquery_path(
                    q, kv_cache, attn_metadata, k_current=k, v_current=v)
            else:
                attn_out = self._verify_decode_path(q, kv_cache, attn_metadata)
        elif attn_metadata.num_decodes == 0:
            if attn_metadata.has_cached_multiquery:
                # Speculative-decode verify (or chunked-prefill continuation):
                # the query tokens have cached history that must be attended.
                # _prefill_first_chunk would drop it; use the context-aware path.
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._cached_multiquery_path(
                    q, kv_cache, attn_metadata, k_current=k, v_current=v)
            else:
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._prefill_first_chunk(q, k, v, attn_metadata, kv_cache)
        else:
            # Mixed batch — split into decode + prefill portions, same as TurboQuant.
            attn_out = self._mixed_batch_path(
                q, key[:N], value[:N], kv_cache, attn_metadata
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ── attention sub-paths ──────────────────────────────────────────────────

    def _flash_varlen(
        self, q, k, v, cu_q, cu_k, max_q, max_k,
    ) -> torch.Tensor:
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=max_q, max_seqlen_k=max_k,
                softmax_scale=self.scale, causal=True,
            )
        return flash_attn_varlen_func(
            q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            softmax_scale=self.scale, causal=True, fa_version=self.fa_version,
        )

    def _prefill_first_chunk(
        self, q, k, v, attn_metadata: KVarNMetadata, kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        """First-chunk prefill: every request's full prompt is in the current
        batch, so attention runs on raw K/V. The K/V have already been
        written to the cache by `do_kv_cache_update`."""
        # Prefer FlashAttention where this ROCm stack supports it. Head-dim 512
        # and images without flash-attn use vLLM's in-tree Triton kernel below.
        if _HAS_FLASH_ATTN and self.head_size <= 256:
            return self._flash_varlen(
                q, k, v,
                cu_q=attn_metadata.query_start_loc,
                cu_k=attn_metadata.query_start_loc,
                max_q=attn_metadata.max_query_len,
                max_k=attn_metadata.max_query_len,
            )
        # vLLM's in-tree Triton prefill kernel accepts raw varlen Q/K/V and
        # writes into caller-owned output. On ROCm/gfx906 this is the normal
        # fallback when upstream flash-attn is unavailable or capped by head
        # size. It writes into caller-owned output, so KV-cache sizing does not
        # need to reserve for hidden attention workspaces.
        if self._prefill_out_buf is None or self._prefill_out_buf.shape[0] < q.shape[0]:
            raise RuntimeError(
                "KVarN static prefill output buffer too small: "
                f"need {q.shape[0]}, have "
                f"{0 if self._prefill_out_buf is None else self._prefill_out_buf.shape[0]}. "
                "Increase scheduler max_num_batched_tokens before startup.")
        out = self._prefill_out_buf[:q.shape[0]]
        context_attention_fwd(
            q=q,
            k=k,
            v=v,
            o=out,
            b_start_loc=attn_metadata.query_start_loc,
            b_seq_len=attn_metadata.seq_lens,
            max_input_len=attn_metadata.max_query_len,
            is_causal=True,
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window,
            sliding_window_k=0,
            # gfx906 has 64 KiB LDS. The generic Triton prefill default
            # BLOCK=64 overflows shared memory at head_size=512; BLOCK=32 keeps
            # the same streaming attention algorithm inside the hardware limit.
            block_size=32 if self.head_size > 256 else None,
        )
        return out[:q.shape[0]].to(q.dtype)

    def _decode_path(
        self, q: torch.Tensor, kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Triton-driven decode: in-kernel dequant + scoring + weighted V,
        with the in-progress fp16 tail buffers combined via LSE in PyTorch.

        Assumes one query token per request (the standard decode regime).
        Mixed-query-length decode steps (e.g. speculative decoding) route to
        the cached multi-query path.
        """
        # If every request contributes exactly one query token, the Triton
        # kernel's (B, Hq) launch shape is valid; otherwise fall back.
        # Use the precomputed Python-int max_query_len (NOT a GPU reduction +
        # host branch — that would force a sync, forbidden during CUDA graph
        # capture).
        if attn_metadata.max_query_len > 1:
            return self._cached_multiquery_path(q, kv_cache, attn_metadata)

        # q shape: [num_decode_tokens, num_heads, head_dim]
        # num_decode_tokens == B (one token per request)
        return kvarn_decode_attention(
            query=q,
            kv_cache=kv_cache,
            hadamard=self._hadamard(q.device),
            scale=self.scale,
            cfg=self.kvarn_config,
            impl=self,
            md=attn_metadata,
        )

    def _verify_decode_path(
        self, q: torch.Tensor, kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Spec-as-decode verify using the builder's persistent vq plan.

        Capture-safe: the vq buffers are persistent (filled CPU-side in
        build() between replays), the block-table/seq-len bound is the
        deployment constant, and the driver's intermediates are created
        inside the captured region (graph-pool managed) — same pattern as
        the single-token fused decode.
        """
        md = attn_metadata
        group = self.kvarn_config.group
        max_ctx_blocks = max((self._max_model_len + group - 1) // group, 1)
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            kvarn_verify_attention,
        )
        B = md.block_table.shape[0]
        return kvarn_verify_attention(
            q, kv_cache, md.block_table, self.scale, self.kvarn_config,
            self, md.vq_req, md.vq_seqlen, max_ctx_blocks,
            qlen=md.vq_qlen, seq_lens=md.seq_lens[:B],
        )

    def _fused_verify_path(
        self, q: torch.Tensor, kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Speculative-decode verify via the fused dual-source kernel.

        Each query token becomes a virtual kernel row with its own
        bottom-right causal length (cached_len + idx + 1) and an indirection
        to its request's block-table row — so the verify step reads int4
        tiles + the fp16 pool directly instead of materializing the whole
        context to fp16 scratch every step (O(context)/step; the issue #10
        long-context MTP collapse: measured 88 -> 45 tok/s from 2K -> 32K
        with the materialize route vs near-flat without MTP).

        Eager-only (verify steps are not graph-captured): fresh small
        tensors per call are fine.
        """
        md = attn_metadata
        B = md.block_table.shape[0]
        n_tok = q.shape[0]
        device = q.device
        group = self.kvarn_config.group

        qsl = md.query_start_loc[:B + 1].to(torch.long)
        qlens = qsl[1:] - qsl[:-1]                              # [B]
        vq_req_long = torch.repeat_interleave(
            torch.arange(B, device=device), qlens)              # [n_tok]
        pos_in_req = torch.arange(n_tok, device=device) - qsl[:-1][vq_req_long]
        committed = md.seq_lens[:B].to(torch.long) - qlens
        vq_seqlen = (committed[vq_req_long] + pos_in_req + 1).to(torch.int32)
        vq_req = vq_req_long.to(torch.int32)

        max_ctx_blocks = min(
            (int(md.max_seq_len) + group - 1) // group,
            md.block_table.shape[1])
        max_ctx_blocks = max(max_ctx_blocks, 1)

        from vllm.v1.attention.ops.triton_kvarn_decode import (
            kvarn_verify_attention,
        )
        return kvarn_verify_attention(
            q, kv_cache, md.block_table, self.scale, self.kvarn_config,
            self, vq_req, vq_seqlen, max_ctx_blocks,
        )

    def _validate_materialized_attn_backend(self) -> str:
        backend = self._materialized_attn_backend
        if backend == "TRITON_ATTN":
            return backend
        if backend == "FLASH_ATTN":
            if not _HAS_FLASH_ATTN:
                raise RuntimeError(
                    "KVARN_MATERIALIZED_ATTN_BACKEND=FLASH_ATTN was selected, "
                    "but flash_attn_varlen_func is not available in this image.")
            if self.head_size > 256:
                raise RuntimeError(
                    "KVARN_MATERIALIZED_ATTN_BACKEND=FLASH_ATTN does not support "
                    f"head_size={self.head_size} on this ROCm stack; use "
                    "TRITON_ATTN for head_size > 256.")
            return backend
        if backend in {"ROCM_AITER_FA", "ROCM_AITER_UNIFIED_ATTN"}:
            raise RuntimeError(
                f"KVARN_MATERIALIZED_ATTN_BACKEND={backend} is not supported "
                "inside KVarN: AITER backends expect their own KV-cache layout "
                "and metadata, not KVarN's materialized dense scratch.")
        raise RuntimeError(
            "Unsupported KVARN_MATERIALIZED_ATTN_BACKEND="
            f"{self._materialized_attn_backend!r}. Supported values: "
            "TRITON_ATTN, FLASH_ATTN.")

    def _cached_multiquery_path(
        self, q: torch.Tensor, kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
        k_current: torch.Tensor | None = None,
        v_current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-query tokens with cached history (a speculative-decode verify
        step or a chunked-prefill continuation), batched (issue #10).

        Builds the batch's rotated fp16 K/V with the ONE block_table-driven
        Triton kernel (``_kvarn_build_packed_kv_kernel``) and runs a
        materialized attention backend. The vLLM Triton continuation kernel
        offsets causal positions by ``seq_len - query_len``, so query token
        ``t`` attends keys ``<= cached_len + t`` — exactly the spec-verify /
        continuation semantics.

        Replaces the old per-request Python gather route. That route did
        per-block host synchronizations, Python dequant, and a large transient
        attention workspace per layer/step, which made MTP decode unusably
        slow and inflated profiling memory enough to collapse KV-cache
        capacity.
        """
        md = attn_metadata
        B = md.block_table.shape[0]
        # Small-qlen multi-query (the spec-decode verify step: every decode
        # step under MTP) goes to the FUSED verify kernel: per-token virtual
        # rows over the dual-source decode kernel, no fp16 materialization.
        # Routed by CONTEXT depth (measured, Qwen3.6-27B AWQ single stream):
        # materialize wins short context (88 vs 81 tok/s @2K — its one
        # write+read is cheap there and the fused per-call overhead shows);
        # fused wins long context (51 vs 45 @32K, growing with depth — the
        # materialize round-trip is the O(context)/step issue #10 MTP
        # slowdown). Crossover ~12K; default threshold 64 blocks (8K).
        # The materialized attention route also keeps LARGE qlen (chunked-prefill
        # continuations), where one materialization amortizes over thousands
        # of query tokens. KVARN_FUSED_VERIFY=0 forces materialize always.
        _group = self.kvarn_config.group
        if (not md.has_transient_query_kv
                and os.environ.get("KVARN_FUSED_VERIFY", "1") == "1"
                and md.max_query_len
                <= int(os.environ.get("KVARN_FUSED_VERIFY_MAXQ", "8"))
                and (int(md.max_seq_len) + _group - 1) // _group
                >= int(os.environ.get("KVARN_FUSED_VERIFY_MIN_BLOCKS", "64"))
                and B > 0):
            return self._fused_verify_path(q, kv_cache, md)
        if self._fa_K_buf is None or self._fa_V_buf is None:
            raise RuntimeError(
                "KVarN cached multi-query requires packed K/V scratch; "
                "scratch buffers were not initialized.")
        backend = self._validate_materialized_attn_backend()

        seq_lens = md.seq_lens[:B].to(torch.int32)
        build_seq_lens = (
            md.fa_build_seq_lens[:B]
            if md.fa_build_seq_lens is not None else seq_lens)
        cu_k = md.fa_cu_seqlens_k
        if cu_k is None:
            raise RuntimeError(
                "KVarN cached multi-query requires precomputed K cu_seqlens; "
                "metadata did not provide fa_cu_seqlens_k.")
        total_k = md.fa_total_k
        if total_k <= 0 or total_k > self._fa_K_buf.shape[0]:
            raise RuntimeError(
                "KVarN cached multi-query packed K/V scratch is too small: "
                f"need {total_k} tokens, have {self._fa_K_buf.shape[0]}. "
                "Reduce max_num_seqs/max_model_len for this run or increase "
                "the KVarN packed-KV scratch cap.")
        if self._prefill_out_buf is None or self._prefill_out_buf.shape[0] < q.shape[0]:
            raise RuntimeError(
                "KVarN cached multi-query Triton output buffer too small: "
                f"need {q.shape[0]}, have "
                f"{0 if self._prefill_out_buf is None else self._prefill_out_buf.shape[0]}. "
                "Increase scheduler max_num_batched_tokens before startup.")

        cfg = self.kvarn_config
        group = cfg.group
        D = self.head_size
        Hk = self.num_kv_heads
        max_k = int(md.max_seq_len)
        max_blocks = min((max_k + group - 1) // group, md.block_table.shape[1])
        max_blocks = max(max_blocks, 1)

        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_build_packed_kv_kernel,
        )

        K_packed = self._fa_K_buf
        V_packed = self._fa_V_buf
        _kvarn_build_packed_kv_kernel[(B * max_blocks, Hk)](
            md.block_table, build_seq_lens, cu_k,
            self._block_to_slot_t,
            kv_cache, self._tail_K_pool, self._tail_V_pool,
            K_packed, V_packed,
            md.block_table.stride(0),
            kv_cache.stride(0), kv_cache.stride(1),
            self._tail_K_pool.stride(0), self._tail_K_pool.stride(1),
            self._tail_K_pool.stride(2),
            K_packed.stride(0), K_packed.stride(1),
            MAX_BLOCKS_PER_REQ=max_blocks,
            D=D, GROUP=group,
            K_BITS=cfg.key_bits, V_BITS=cfg.value_bits,
            NUM_BLOCKS_LOOKUP=self._block_lookup_size,
            K_PACKED_OFFSET=cfg.k_packed_offset, K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset, K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset, V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset, V_ZP_OFFSET=cfg.v_zp_offset,
            num_warps=4, num_stages=2,
        )

        # The packed K/V are in the rotated frame (the store path rotates before
        # quantizing / pooling), so rotate q in and un-rotate the output — same
        # fp16 Hadamard as the store side, so QK^T is invariant.
        H16 = (self._H_fp16 if self._H_fp16 is not None
               else self._hadamard(q.device).to(torch.float16))
        n_tok = q.shape[0]
        if md.has_transient_query_kv:
            if k_current is None or v_current is None:
                raise RuntimeError(
                    "KVarN transient query K/V requested without current K/V.")
            if (md.query_start_locs_cpu is None or md.query_lens_cpu is None
                    or md.seq_lens_cpu is None
                    or md.transient_query_kv_rows_cpu is None):
                raise RuntimeError(
                    "KVarN transient query K/V metadata is incomplete.")
            if self._k_rot_scratch is None or self._v_rot_scratch is None:
                raise RuntimeError(
                    "KVarN rotation scratch was not initialized.")
            k_rot_cur = self._k_rot_scratch[:n_tok]
            v_rot_cur = self._v_rot_scratch[:n_tok]
            torch.matmul(k_current[:n_tok].to(torch.float16), H16, out=k_rot_cur)
            torch.matmul(v_current[:n_tok].to(torch.float16), H16, out=v_rot_cur)
            if md.fa_cu_seqlens_k_cpu is None:
                raise RuntimeError(
                    "KVarN transient query K/V metadata is missing CPU K offsets.")
            cu_k_cpu = md.fa_cu_seqlens_k_cpu
            for b in range(B):
                if not md.transient_query_kv_rows_cpu[b]:
                    continue
                q_start = md.query_start_locs_cpu[b]
                q_len = md.query_lens_cpu[b]
                committed = max(md.seq_lens_cpu[b] - q_len, 0)
                dst = cu_k_cpu[b] + committed
                K_packed[dst:dst + q_len].copy_(
                    k_rot_cur[q_start:q_start + q_len])
                V_packed[dst:dst + q_len].copy_(
                    v_rot_cur[q_start:q_start + q_len])

        flat_rows = n_tok * self.num_heads
        if (self._materialized_q_rot_buf is None
                or self._materialized_out_buf is None
                or self._materialized_q_rot_buf.shape[0] < flat_rows
                or self._materialized_out_buf.shape[0] < flat_rows):
            raise RuntimeError(
                "KVarN materialized attention rotation scratch too small: "
                f"need {flat_rows} rows.")
        q_rot_flat = self._materialized_q_rot_buf[:flat_rows]
        torch.mm(q.reshape(-1, D).to(torch.float16), H16, out=q_rot_flat)
        q_rot = q_rot_flat.view(n_tok, self.num_heads, D)

        if backend == "FLASH_ATTN":
            out_rot = self._flash_varlen(
                q_rot, K_packed[:total_k], V_packed[:total_k],
                cu_q=md.query_start_loc[:B + 1],
                cu_k=cu_k,
                max_q=md.max_query_len,
                max_k=max_k,
            )
        else:
            out_rot = self._prefill_out_buf[:n_tok]
            context_attention_fwd_with_kv_lens(
                q=q_rot,
                k=K_packed[:total_k],
                v=V_packed[:total_k],
                o=out_rot,
                q_start_loc=md.query_start_loc[:B + 1],
                k_start_loc=cu_k,
                seq_lens=seq_lens,
                max_query_len=md.max_query_len,
                is_causal=True,
                softmax_scale=self.scale,
                sliding_window_q=self.sliding_window,
                sliding_window_k=0,
                block_size=32 if self.head_size > 256 else None,
            )
        out_flat = self._materialized_out_buf[:flat_rows]
        torch.mm(out_rot.reshape(-1, D), H16, out=out_flat)
        return out_flat.view(n_tok, self.num_heads, D)

    def _mixed_batch_path(
        self, q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor,
        kv_cache: torch.Tensor, attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Split mixed batch into decode-then-prefill, mirroring TurboQuant."""
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        N = attn_metadata.num_actual_tokens

        out = torch.empty(N, self.num_heads, self.head_size,
                          dtype=q.dtype, device=q.device)

        # Build the Stage α-2 fa_* fields for the decode subset. This path
        # always runs eager (mixed batches aren't graph-captured), so fresh
        # (non-persistent) tensors are fine.
        group = self.kvarn_config.group
        dec_seq_lens = attn_metadata.seq_lens[:num_decodes].to(torch.int32)
        dec_cu_k = (
            attn_metadata.fa_cu_seqlens_k[:num_decodes + 1]
            if attn_metadata.fa_cu_seqlens_k is not None
            else None
        )
        dec_cu_q = (
            attn_metadata.fa_cu_seqlens_q[:num_decodes + 1]
            if attn_metadata.fa_cu_seqlens_q is not None
            else None
        )
        dec_transient_rows = (
            attn_metadata.transient_query_kv_rows_cpu[:num_decodes]
            if attn_metadata.transient_query_kv_rows_cpu is not None else None)
        mbpr = (self._max_model_len + group - 1) // group
        decode_meta = KVarNMetadata(
            seq_lens=attn_metadata.seq_lens[:num_decodes],
            slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
            block_table=attn_metadata.block_table[:num_decodes],
            query_start_loc=attn_metadata.query_start_loc[:num_decodes + 1],
            num_actual_tokens=num_decode_tokens,
            max_query_len=1, max_seq_len=attn_metadata.max_seq_len,
            is_prefill=False,
            fa_cu_seqlens_q=dec_cu_q,
            fa_cu_seqlens_k=dec_cu_k,
            fa_cu_seqlens_k_cpu=(
                attn_metadata.fa_cu_seqlens_k_cpu[:num_decodes + 1]
                if attn_metadata.fa_cu_seqlens_k_cpu is not None else None),
            fa_build_seq_lens=(
                attn_metadata.fa_build_seq_lens[:num_decodes]
                if attn_metadata.fa_build_seq_lens is not None else None),
            fa_max_blocks_per_req=mbpr,
            fa_max_seqlen_k_fixed=self._max_model_len,
            seq_lens_cpu=(
                attn_metadata.seq_lens_cpu[:num_decodes]
                if attn_metadata.seq_lens_cpu is not None else None),
            query_start_locs_cpu=(
                attn_metadata.query_start_locs_cpu[:num_decodes + 1]
                if attn_metadata.query_start_locs_cpu is not None else None),
            query_lens_cpu=(
                attn_metadata.query_lens_cpu[:num_decodes]
                if attn_metadata.query_lens_cpu is not None else None),
            has_transient_query_kv=bool(dec_transient_rows and any(dec_transient_rows)),
            transient_query_kv_rows_cpu=dec_transient_rows,
        )
        if attn_metadata.vq_seqlen is not None:
            # Spec-as-decode: the decode portion carries multi-token verify
            # queries — use the vq plan (mixed batches run eager, slices ok).
            decode_meta.vq_req = attn_metadata.vq_req[:num_decode_tokens]
            decode_meta.vq_seqlen = attn_metadata.vq_seqlen[:num_decode_tokens]
            decode_meta.vq_qlen = attn_metadata.vq_qlen
            if decode_meta.has_transient_query_kv:
                k_dec = k_all[:num_decode_tokens].view(
                    -1, self.num_kv_heads, self.head_size)
                v_dec = v_all[:num_decode_tokens].view(
                    -1, self.num_kv_heads, self.head_size)
                out[:num_decode_tokens] = self._cached_multiquery_path(
                    q[:num_decode_tokens], kv_cache, decode_meta,
                    k_current=k_dec, v_current=v_dec)
            else:
                out[:num_decode_tokens] = self._verify_decode_path(
                    q[:num_decode_tokens], kv_cache, decode_meta,
                )
        else:
            out[:num_decode_tokens] = self._decode_path(
                q[:num_decode_tokens], kv_cache, decode_meta,
            )

        prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
        prefill_qsl = attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
        prefill_cu_k = (
            attn_metadata.fa_cu_seqlens_k[num_decodes:]
            if attn_metadata.fa_cu_seqlens_k is not None
            else None
        )
        prefill_seq_lens_cpu = (
            attn_metadata.seq_lens_cpu[num_decodes:]
            if attn_metadata.seq_lens_cpu is not None
            else None
        )
        prefill_qsl_cpu = None
        if attn_metadata.query_start_locs_cpu is not None:
            prefill_qsl_cpu = [
                x - num_decode_tokens
                for x in attn_metadata.query_start_locs_cpu[num_decodes:]
            ]
        prefill_transient_rows = (
            attn_metadata.transient_query_kv_rows_cpu[num_decodes:]
            if attn_metadata.transient_query_kv_rows_cpu is not None else None)
        prefill_meta = KVarNMetadata(
            seq_lens=prefill_seq_lens,
            slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
            block_table=attn_metadata.block_table[num_decodes:],
            query_start_loc=prefill_qsl,
            num_actual_tokens=N - num_decode_tokens,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=attn_metadata.max_seq_len,  # WSL fix (PR #16): avoid per-step .item() D2H sync (global max is a safe upper bound for the prefill kernel)
            is_prefill=True,
            seq_lens_cpu=prefill_seq_lens_cpu,
            query_start_locs_cpu=prefill_qsl_cpu,
            query_lens_cpu=(
                attn_metadata.query_lens_cpu[num_decodes:]
                if attn_metadata.query_lens_cpu is not None else None),
            fa_cu_seqlens_k=prefill_cu_k,
            fa_cu_seqlens_k_cpu=(
                attn_metadata.fa_cu_seqlens_k_cpu[num_decodes:]
                if attn_metadata.fa_cu_seqlens_k_cpu is not None else None),
            fa_build_seq_lens=(
                attn_metadata.fa_build_seq_lens[num_decodes:]
                if attn_metadata.fa_build_seq_lens is not None else None),
            fa_total_k=attn_metadata.fa_total_k,
            has_transient_query_kv=bool(
                prefill_transient_rows and any(prefill_transient_rows)),
            transient_query_kv_rows_cpu=prefill_transient_rows,
        )
        if attn_metadata.has_cached_multiquery:
            # The multi-query (prefill-classified) requests here are speculative
            # -decode verify steps / chunked-prefill continuations with cached
            # history — attend over the cached K/V, not just the new tokens.
            k_pref = k_all[num_decode_tokens:].view(
                -1, self.num_kv_heads, self.head_size)
            v_pref = v_all[num_decode_tokens:].view(
                -1, self.num_kv_heads, self.head_size)
            out[num_decode_tokens:] = self._cached_multiquery_path(
                q[num_decode_tokens:], kv_cache, prefill_meta,
                k_current=k_pref, v_current=v_pref)
        else:
            k_pref = k_all[num_decode_tokens:].view(-1, self.num_kv_heads, self.head_size)
            v_pref = v_all[num_decode_tokens:].view(-1, self.num_kv_heads, self.head_size)
            out[num_decode_tokens:] = self._prefill_first_chunk(
                q[num_decode_tokens:], k_pref, v_pref, prefill_meta, kv_cache,
            )
        return out
