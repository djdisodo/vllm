"""Dense KVarN shared-prefix stress (issue #10 loops validation).

A growing multi-turn conversation over a long shared prefix, greedy. Each
round's answer is appended to the history, so every round is a prefix-cache
hit on all earlier blocks (PHASE=cache) or a full recompute (PHASE=nocache).
Saves all answers to JSON; run both phases and diff — correct caching makes
them identical. Also reports a repetition heuristic per round.

Env: MODEL, KV, PHASE (cache|nocache), SPEC (1 -> MTP k=3), ROUNDS (default 8).
"""
import json
import os

from vllm import LLM, SamplingParams


def main():
    M = os.environ["MODEL"]
    KV = os.environ.get("KV", "kvarn_k4v2_g128")
    PHASE = os.environ.get("PHASE", "cache")
    SPEC = os.environ.get("SPEC", "0") == "1"
    ROUNDS = int(os.environ.get("ROUNDS", "8"))

    kw = dict(
        model=M,
        dtype="bfloat16",
        max_model_len=16384,
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
    sp = SamplingParams(temperature=0.0, max_tokens=256)

    # System prompt past one 3200-token manager block (hybrid + prefix caching
    # quantizes cache hits to 3200-token granularity), so every round after the
    # first is a real cross-request cache hit.
    system = (
        "You are a precise assistant. Reference document: "
        + " ".join(
            f"Item {i} is assigned the codename '{c}{i}' and the value {i * 7 % 101}."
            for i, c in zip(range(1, 200), "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10)
        )
    )
    questions = [
        "What is the value of item 5? Answer in one sentence.",
        "What codename does item 12 have? One sentence.",
        "Sum the values of items 3 and 9. One sentence.",
        "Which item has codename 'D4'? One sentence.",
        "Is the value of item 20 even or odd? One sentence.",
        "What is the value of item 33 minus the value of item 1? One sentence.",
        "State item 7's codename and value. One sentence.",
        "What is the value of item 50? One sentence.",
        "Multiply item 2's value by 3. One sentence.",
        "Which has the larger value, item 10 or item 11? One sentence.",
        "What is item 60's codename? One sentence.",
        "Add the values of items 14 and 15. One sentence.",
    ][:ROUNDS]

    # Unrelated conversation interleaved between rounds: its builds trigger
    # the reclaim of the finished round's pool slots (in serving this is just
    # other traffic; back-to-back turns never reclaim and accidentally stay
    # correct).
    other = [{"role": "user", "content": " ".join(
        f"City {i} lies {i * 13 % 97} km from the capital." for i in range(1, 50)
    ) + "\n\nWhich city is closest to the capital? One sentence."}]

    msgs = [{"role": "system", "content": system}]
    answers = []
    for i, q in enumerate(questions):
        text_other = tok.apply_chat_template(
            other, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        llm.generate([text_other], sp)
        msgs.append({"role": "user", "content": q})
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ro = llm.generate([text], sp)[0]
        out = ro.outputs[0]
        a = out.text
        msgs.append({"role": "assistant", "content": a})
        ids = list(out.token_ids)
        tail = ids[-120:]
        uniq = round(len(set(tail)) / max(len(tail), 1), 3)
        cached = getattr(ro, "num_cached_tokens", -1)
        answers.append({"round": i, "q": q, "a": a, "uniq_frac_tail": uniq,
                        "cached_tokens": cached})
        print(f"ROUND {i} cached={cached} uniq={uniq} a={a[:100]!r}")

    out_path = f"/tmp/prefix_stress_{PHASE}_{'spec' if SPEC else 'nospec'}.json"
    with open(out_path, "w") as f:
        json.dump(answers, f, indent=2)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
