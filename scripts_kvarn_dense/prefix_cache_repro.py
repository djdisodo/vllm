"""Dense KVarN prefix-caching corruption repro (issue #10 'loops').

Same engine, greedy. Request 1 runs cold; request 2 is the IDENTICAL prompt,
i.e. a full prefix-cache hit served after request 1 finished and its fp16 pool
slots were reclaimed. With correct caching the two outputs are byte-identical.
Then a multi-turn round: prompt + answer + follow-up (partial cache hit),
compared against the same turn replayed with prefix caching disabled in a
separate engine run (pass PHASE=nocache).

Env: MODEL (path or HF id), KV (kv_cache_dtype or 'auto'), PHASE (cache|nocache),
SPEC (1 to enable MTP num_speculative_tokens=3).
"""
import json
import os

from vllm import LLM, SamplingParams


def main():
    M = os.environ["MODEL"]
    KV = os.environ.get("KV", "kvarn_k4v2_g128")
    PHASE = os.environ.get("PHASE", "cache")
    SPEC = os.environ.get("SPEC", "0") == "1"

    kw = dict(
        model=M,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=8,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enable_prefix_caching=(PHASE == "cache"),
    )
    if KV != "auto":
        kw["kv_cache_dtype"] = KV
        kw["block_size"] = 128
    if SPEC:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 3}

    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=400)

    # Prompt long enough to span the MANAGER block size: hybrid models with
    # prefix caching get 3200-token manager blocks (mamba 'align' page-size
    # reconciliation), and cache hits only happen at that granularity. Aim for
    # ~6800 tokens = 2 full manager blocks (= 50 KVarN kernel tiles).
    filler = " ".join(
        f"Fact {i}: the {i}-th prime squared plus one is even only when the prime is odd."
        for i in range(1, 320)
    )
    question = (
        "\n\nUsing the facts above, explain in a numbered list of exactly five "
        "items why the sum of two odd primes is always even. Be concise."
    )

    def chat(messages):
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ro = llm.generate([text], sp)[0]
        print(f"  [cached_tokens={getattr(ro, 'num_cached_tokens', '?')} "
              f"prompt_len={len(ro.prompt_token_ids)}]")
        return ro.outputs[0].text

    # Unrelated request run BETWEEN turns: its builds trigger the reclaim of
    # the finished turn's pool slots (in serving this is just other traffic;
    # back-to-back generates never reclaim and accidentally stay correct).
    other = [{"role": "user", "content": " ".join(
        f"City {i} lies {i * 13 % 97} km from the capital." for i in range(1, 50)
    ) + "\n\nWhich city is closest to the capital? One sentence."}]

    msgs = [{"role": "user", "content": filler + question}]

    # Turn 1, cold.
    a1 = chat(msgs)
    chat(other)
    # Turn 1 again — full prefix-cache hit on the finished request's blocks,
    # AFTER the filler's builds reclaimed them.
    a1_hit = chat(msgs)
    chat(other)

    # Turn 2 — chat continuation (partial hit on turn-1 blocks).
    msgs2 = msgs + [
        {"role": "assistant", "content": a1},
        {"role": "user", "content": "Now restate item 3 of your list in one sentence."},
    ]
    a2 = chat(msgs2)

    res = {
        "phase": PHASE,
        "kv": KV,
        "spec": SPEC,
        "identical_replay": a1 == a1_hit,
        "a1": a1,
        "a1_hit": a1_hit,
        "a2": a2,
    }
    out_path = f"/tmp/prefix_repro_{PHASE}_{'spec' if SPEC else 'nospec'}_{KV}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("IDENTICAL_REPLAY:", a1 == a1_hit)
    print("A1   HEAD:", repr(a1[:140]))
    print("A1HIT HEAD:", repr(a1_hit[:140]))
    print("A2   HEAD:", repr(a2[:200]))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
