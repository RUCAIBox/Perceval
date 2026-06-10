# Serving the Perceval reward backends

Training a Perceval policy requires two OpenAI-compatible HTTP services running
alongside the trainer:

| Service | Role | Endpoint env var |
| --- | --- | --- |
| LLM-as-judge | Verifies final answers against the ground-truth answer | `LLM_AS_A_JUDGE_BASE` |
| PRM (Process Reward Model) | Localizes perceptual errors inside a response and returns a sub-sentence list | `PRM_BASE` |

Both env vars accept a **space-separated list** of base URLs; the reward
function picks one uniformly at random per request, which makes it trivial to
scale by launching more replicas. The PRM endpoint in particular benefits
from 2-8 replicas because every rollout generates many requests.

## Quick start (single node, 8 GPUs)

```bash
# 1 GPU for the judge
CUDA_VISIBLE_DEVICES=0 PORT=9999 \
  PERCEVAL_JUDGE_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
  bash serve_judge.sh &

# 3 GPUs serving 3 PRM replicas
for i in 0 1 2; do
  port=$((12298 + i))
  CUDA_VISIBLE_DEVICES=$((i + 1)) PORT=$port \
    PERCEVAL_PRM_MODEL_PATH=/path/to/perceval-prm \
    bash serve_prm.sh &
done
wait
```

In `configs/perceval.env`, then point the trainer at the URLs:

```bash
LLM_AS_A_JUDGE_BASE="http://127.0.0.1:9999/v1"
PRM_BASE="http://127.0.0.1:12298/v1 http://127.0.0.1:12299/v1 http://127.0.0.1:12300/v1"
```

## Multi-node

Bind each replica to `0.0.0.0` (already the default in the scripts) and put the
node's reachable IP in the env var instead of `127.0.0.1`. The trainer treats
the list as flat - it doesn't care which node a URL points to.

## Health check

```bash
curl -s http://127.0.0.1:12298/v1/models | jq
```

Should return a JSON object whose `data[0].id` matches `--served-model-name`
(`judge` or `prm`). The trainer fetches this on startup to discover model
names; if the model list is empty, requests will fail later with a confusing
error - check the server log.
