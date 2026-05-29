[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/huawei-csl/KVarN?label=Stars&logo=github&logoColor=white&style=flat-square)](https://github.com/huawei-csl/KVarN/stargazers)
[![hf-space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Huawei%20CSL-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/huawei-csl)
[![Built on vLLM](https://img.shields.io/badge/Built%20on-vLLM%20v0.22.0-30a14e)](https://github.com/vllm-project/vllm)



<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td><img src="imgs/logo_600.png" alt="KVarN Logo" width="160"></td>
    <td style="vertical-align: middle;"><h1>KVarN: Variance-Normalized KV-Cache Quantization for vLLM</h1></td>
  </tr>
</table>



> ⚡️ **Near-lossless KV-cache compression for vLLM.** Fit far longer contexts, sustain higher long-context throughput, and keep FP16-level accuracy.

> 💡 **Want longer context or more concurrent requests on the same GPU?** KVarN shrinks the KV cache by 3-5x while preserving model quality, even on reasoning models.

> 🔌 **Drop-in.** It is a native vLLM attention backend: add one flag, no model changes, no calibration.

---

## Quickstart

KVarN ships as a vLLM fork. Install it like vLLM, then select the KVarN KV-cache dtype.

```bash
# 1. Clone
git clone https://github.com/huawei-csl/KVarN.git
cd KVarN

# 2. Install (uses the upstream precompiled wheel; KVarN kernels are Triton, JIT-compiled at runtime)
VLLM_USE_PRECOMPILED=1 pip install -e .
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-32B",
    kv_cache_dtype="kvarn_k4v2_g128",   # enable KVarN
    block_size=128,                     # KVarN tile size
)
print(llm.generate("Explain KV-cache quantization in one sentence.",
                    SamplingParams(max_tokens=64))[0].outputs[0].text)
```

Serving works the same way:

```bash
vllm serve Qwen/Qwen3-32B --kv-cache-dtype kvarn_k4v2_g128 --block-size 128
```

---

## How does KVarN work?

KVarN quantizes the KV cache one fixed-size token tile at a time, in three steps:

1. **Hadamard rotation** along the channel dimension. This mixes channels so that
   per-channel outliers are spread out, making the tile easier to quantize. The
   rotation is orthonormal, so attention scores are preserved.

2. **Iterative variance normalization** (Sinkhorn-like). Alternating column-wise
   and row-wise standard-deviation normalization in log space equalizes the
   variance across rows and columns of the tile. Balancing the tile this way
   shrinks the quantization error before any rounding happens.

3. **Asymmetric round-to-nearest** at low bit-width, with the scales folded back
   in at read time. Keys are quantized per channel, values per token.

The shipped preset spends **more bits on keys than values** (`kvarn_k4v2_g128`:
4-bit keys, 2-bit values). The reason is structural: key error propagates through
the `softmax(QK^T)` exponentials, while value error is averaged out by the softmax
weights, so keys carry most of the quantization sensitivity. The bit-widths are
fully parameterized internally, so other presets are easy to add.

---

## Roadmap

- Additional bit-width presets (the quantizer and kernels are bit-width generic).
- Variable page size (tile sizes other than 128).
- Broader model coverage and benchmarks.

---

## License and attribution

KVarN is built on [vLLM](https://github.com/vllm-project/vllm) (v0.22.0) and is
released under the Apache 2.0 License. The original vLLM README is preserved as
[`README_vLLM.md`](README_vLLM.md).
