#!/bin/bash
# Dense KVarN prefix-caching fix validation battery (issue #10 loops).
# Runs sequentially on one GPU; each phase writes /tmp/prefix_*.json + a log.
set -u
cd /mnt/nvme1/KVarN
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_CACHE=/mnt/nvme1/huggingface
export MODEL=/mnt/nvme1/huggingface/models--cyankiwi--Qwen3.6-27B-AWQ-BF16-INT4/snapshots/232ed78ba2a5eba6c39cd36d1e11ff7347e0883e
export KV=kvarn_k4v2_g128
PY=.venv/bin/python

run() {  # run <tag> <script> <env overrides...>
  local tag=$1 script=$2; shift 2
  echo "=== RUN $tag ==="
  env "$@" $PY "scripts_kvarn_dense/$script" > "/tmp/val_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] && grep -q "Engine core initialization failed" "/tmp/val_${tag}.log"; then
    echo "(engine init failed — retrying once after 20s)"
    sleep 20
    env "$@" $PY "scripts_kvarn_dense/$script" > "/tmp/val_${tag}.log" 2>&1
    rc=$?
  fi
  echo "EXIT_${tag}=$rc"
  grep -E "IDENTICAL_REPLAY|ROUND|saved" "/tmp/val_${tag}.log" | tail -15
}

run repro_cache      prefix_cache_repro.py PHASE=cache  SPEC=0
run repro_nocache    prefix_cache_repro.py PHASE=nocache SPEC=0
run stress_cache     prefix_stress.py      PHASE=cache  SPEC=0
run stress_nocache   prefix_stress.py      PHASE=nocache SPEC=0
run repro_cache_mtp  prefix_cache_repro.py PHASE=cache  SPEC=1
run stress_cache_mtp prefix_stress.py      PHASE=cache  SPEC=1
run stress_nocache_mtp prefix_stress.py    PHASE=nocache SPEC=1

echo "=== DIFFS ==="
$PY - <<'EOF'
import json
def load(p):
    return json.load(open(p))
# repro: cache-hit replay identical + cold answers match nocache reference
rc = load('/tmp/prefix_repro_cache_nospec_kvarn_k4v2_g128.json')
rn = load('/tmp/prefix_repro_nocache_nospec_kvarn_k4v2_g128.json')
print('repro identical_replay (cache):', rc['identical_replay'])
print('repro cold a1 cache==nocache:', rc['a1'] == rn['a1'])
print('repro turn2 a2 cache==nocache:', rc['a2'] == rn['a2'])
sc = load('/tmp/prefix_stress_cache_nospec.json')
sn = load('/tmp/prefix_stress_nocache_nospec.json')
m = [i for i, (a, b) in enumerate(zip(sc, sn)) if a['a'] != b['a']]
print(f'stress rounds mismatched (cache vs nocache): {m or "none"} / {len(sc)}')
scm = load('/tmp/prefix_stress_cache_spec.json')
snm = load('/tmp/prefix_stress_nocache_spec.json')
mm = [i for i, (a, b) in enumerate(zip(scm, snm)) if a['a'] != b['a']]
print(f'stress+MTP rounds mismatched: {mm or "none"} / {len(scm)}')
rcm = load('/tmp/prefix_repro_cache_spec_kvarn_k4v2_g128.json')
print('repro+MTP identical_replay:', rcm['identical_replay'])
EOF
echo "=== DONE ==="
